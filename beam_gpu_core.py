"""
BeamSearcher の GPU(CUDA) 高速化版コア実装。

設計方針:
  - 各レベルで (beam_size x next_p) 個の (AND + popcount) は完全に独立 -> GPU向き
  - 全候補の next_mask をフルに作らず、まず「popcount(スコア)だけ」を全候補について計算
    (int32を1個/候補なので軽い)。
  - 上位 beam_width 件を選んだ後に、その分だけ next_mask を作る(この時だけ AND を再計算)。
    -> メモリ使用量を (beam_size x next_p x n_words) ではなく (beam_size x next_p) に抑える。
  - パスは「親のビーム内インデックス + 自分のシフトs」だけを各レベルで保持し、
    最後にまとめて辿って復元する(可変長リストのコピーを毎ノードで行わない)。

CPUBackend(numpy) と GPUBackend(cupy RawKernel) は同一インターフェースを持ち、
アルゴリズム全体 (BeamSearcherGPU) はバックエンドを差し替えるだけで
CPUでも(検証用として)、GPUでも(本番用として)動く。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Union, Callable, Optional
import numpy as np
from tqdm import tqdm
import Config as cfg

logger = logging.getLogger(__name__)

BeamWidthSpec = Union[int, List[int], Callable[[int], int]]


# ----------------------------------------------------------------------
# ビットテーブル構築 (numpy版・ベクトル化)
# ----------------------------------------------------------------------
def build_bit_tables_packed(primes: List[int], cols: int) -> Tuple[List[np.ndarray], int]:
    """
    各 (p, s) のビットマスクを uint64 配列 (n_words,) にパックしたテーブルを構築する。
    戻り値: (tables, n_words)
        tables[level] は shape (p, n_words) の numpy uint64 配列
    元の HLSearch.build_bit_tables (bigint版) とビット位置の対応は完全に一致する
    (verify_packing.py で検証済み)。
    """
    n_words = (cols + 63) // 64
    j = np.arange(cols, dtype=np.int64)
    tables = []
    for p in primes:
        residues = j % p
        s_arr = np.arange(p, dtype=np.int64)[:, None]
        keep = residues[None, :] != s_arr  # (p, cols) bool

        packed = np.zeros((p, n_words), dtype=np.uint64)
        for w in range(n_words):
            lo = w * 64
            hi = min(lo + 64, cols)
            if lo >= cols:
                break
            chunk = keep[:, lo:hi].astype(np.uint64)
            weights = np.uint64(1) << np.arange(hi - lo, dtype=np.uint64)
            packed[:, w] = (chunk * weights).sum(axis=1, dtype=np.uint64)
        tables.append(packed)
    return tables, n_words


def popcount64_numpy(x: np.ndarray) -> np.ndarray:
    """uint64配列の要素ごとpopcount (SWARアルゴリズム、numpyベクトル化)"""
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return x.astype(np.int64)


# ----------------------------------------------------------------------
# バックエンド: CPU(numpy) 版 (正当性検証・GPUなし環境でのフォールバック用)
# ----------------------------------------------------------------------
class CPUBackend:
    name = "cpu(numpy)"

    def __init__(self, tables: List[np.ndarray], n_words: int):
        self.tables = tables
        self.n_words = n_words

    def score_all(self, beam_masks: np.ndarray, level: int) -> np.ndarray:
        """
        beam_masks: (B, n_words) uint64
        戻り値: (B, P) int64 の popcount(beam_masks[b] & table[level][s])
        """
        table = self.tables[level]  # (P, n_words)
        # (B, 1, n_words) & (1, P, n_words) -> (B, P, n_words)
        anded = beam_masks[:, None, :] & table[None, :, :]
        counts = popcount64_numpy(anded).sum(axis=-1)
        return counts

    def gather_masks(self, beam_masks: np.ndarray, level: int,
                      b_idx: np.ndarray, s_idx: np.ndarray) -> np.ndarray:
        table = self.tables[level]
        return beam_masks[b_idx] & table[s_idx]


# ----------------------------------------------------------------------
# バックエンド: GPU(CUDA/CuPy) 版
# ----------------------------------------------------------------------
_CUDA_KERNEL_SRC = r"""
extern "C" __global__
void score_kernel(
    const unsigned long long* __restrict__ beam_masks, // (B, n_words)
    const unsigned long long* __restrict__ table,       // (P, n_words)
    long long* __restrict__ out_counts,                 // (B, P)
    int n_words, int P, int B)
{
    extern __shared__ unsigned long long shared_beam[];
    int b = blockIdx.x;
    if (b >= B) return;

    // beam_masks[b] を shared memory にロード(このブロック内の全スレッドで再利用)
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        shared_beam[w] = beam_masks[(size_t)b * n_words + w];
    }
    __syncthreads();

    for (int s = blockIdx.y * blockDim.x + threadIdx.x; s < P; s += gridDim.y * blockDim.x) {
        const unsigned long long* row = table + (size_t)s * n_words;
        long long cnt = 0;
        for (int w = 0; w < n_words; ++w) {
            cnt += __popcll(shared_beam[w] & row[w]);
        }
        out_counts[(size_t)b * P + s] = cnt;
    }
}
"""


class GPUBackend:
    name = "cuda(cupy)"

    def __init__(self, tables_np: List[np.ndarray], n_words: int, block_size: int = 128):
        import cupy as cp  # 遅延import: cupy未導入環境でもCPUBackendは使えるようにする
        self.cp = cp
        self.n_words = n_words
        self.block_size = block_size
        # 各レベルのテーブルを常時GPU上に保持
        self.tables = [cp.asarray(t) for t in tables_np]
        self._kernel = cp.RawKernel(_CUDA_KERNEL_SRC, "score_kernel")

    def score_all(self, beam_masks, level: int):
        """
        beam_masks: cupy (B, n_words) uint64
        戻り値: cupy (B, P) int64
        """
        cp = self.cp
        table = self.tables[level]
        B = beam_masks.shape[0]
        P = table.shape[0]
        out = cp.empty((B, P), dtype=cp.int64)

        block = self.block_size
        grid_y = max(1, (P + block - 1) // block)
        shared_bytes = self.n_words * 8  # unsigned long long

        self._kernel(
            (B, grid_y), (block,),
            (beam_masks, table, out, np.int32(self.n_words), np.int32(P), np.int32(B)),
            shared_mem=shared_bytes,
        )
        return out

    def gather_masks(self, beam_masks, level: int, b_idx, s_idx):
        table = self.tables[level]
        return beam_masks[b_idx] & table[s_idx]


# ----------------------------------------------------------------------
# BeamSearcher (GPU/CPU共通ロジック)
# ----------------------------------------------------------------------
class BeamSearcherGPU:
    """
    元の BeamSearcher と同じ探索アルゴリズム(ビーム法)を、
    backend (CPUBackend / GPUBackend) を通して並列実行する版。

    backend が GPUBackend なら CUDA で、CPUBackend なら numpy でスコア計算する。
    探索ロジック自体(枝刈り条件・打ち切り条件)はオリジナルと同一。
    """

    def __init__(
        self,
        primes: List[int],
        depth: int,
        limit: int,
        target: int,
        max_depth: int,
        cols: int,
        output_file: Path,
        beam_width: BeamWidthSpec = 100,
        save_all_best: bool = False,
        use_gpu: bool = True,
    ):
        self.primes = primes[:depth]
        self.depth = depth
        self.limit = limit
        self.target = target
        self.max_depth = max_depth
        self.cols = cols
        self.output_file = output_file
        self.beam_width = beam_width
        self.save_all_best = save_all_best

        tables_np, n_words = build_bit_tables_packed(self.primes, cols)
        self.n_words = n_words

        self.backend = None
        if use_gpu:
            try:
                self.backend = GPUBackend(tables_np, n_words)
            except ImportError:
                logger.warning("cupy が見つからないため CPU(numpy) backend にフォールバックします")
        if self.backend is None:
            self.backend = CPUBackend(tables_np, n_words)

        self.xp = self.backend.cp if isinstance(self.backend, GPUBackend) else np

        self.max_count = 0
        self.results = 0
        self.nodes_searched = 0
        self.best_paths: List[List[int]] = []

    def _resolve_beam_width(self, level: int) -> int:
        bw = self.beam_width
        if callable(bw):
            width = bw(level)
        elif isinstance(bw, (list, tuple)):
            width = bw[level] if level < len(bw) else bw[-1]
        else:
            width = bw
        if width < 1:
            raise ValueError(f"beam_width は1以上である必要があります (level={level}, width={width})")
        return width

    def run(self) -> "BeamSearcherGPU":
        import time
        xp = self.xp
        start_time = time.time()

        first_p = self.primes[0]
        # level 0 の初期マスク = 全ビット1 なので table[0] そのものが next_mask
        table0 = self.backend.tables[0]
        counts0 = self._row_popcount(table0)  # (first_p,) 各行の合計popcount

        self.nodes_searched += first_p
        valid = counts0 >= self.limit
        idx_valid = xp.where(valid)[0]

        # (score, mask, parent=-1, shift)
        scores = counts0[idx_valid]
        masks = table0[idx_valid]
        shifts = idx_valid

        beam_scores, beam_masks, beam_shifts = self._select_topk(scores, masks, shifts, self._resolve_beam_width(0))
        parents_per_level: List[np.ndarray] = [np.full(len(beam_shifts), -1, dtype=np.int64)]
        shifts_per_level: List[np.ndarray] = [self._to_cpu(beam_shifts)]
        if cfg.SHOW_PROGRESS:
            pbar = tqdm(
                total=self.depth,
                desc="Searching...",
                unit="Task",
                dynamic_ncols=True,
            )

        for level in range(1, self.depth):
            if cfg.SHOW_PROGRESS:
                pbar.update(level)
            B = beam_masks.shape[0]
            P = self.primes[level]
            is_last_level = (level + 1 >= self.depth)

            all_counts = self.backend.score_all(beam_masks, level)  # (B, P)
            self.nodes_searched += B * P

            flat = all_counts.reshape(-1)
            valid_mask = flat >= self.limit
            if is_last_level and self.depth == self.max_depth:
                valid_mask &= flat <= self.target

            flat_idx = self._nonzero(valid_mask)
            if flat_idx.shape[0] == 0:
                logger.warning("ビームが空になりました (level=%d)。探索を打ち切ります。", level)
                break

            flat_scores = flat[flat_idx]
            width = self._resolve_beam_width(level)
            top_local = self._topk_indices(flat_scores, width)
            sel_flat_idx = flat_idx[top_local]
            sel_scores = flat_scores[top_local]

            b_idx = sel_flat_idx // P
            s_idx = sel_flat_idx % P

            next_masks = self.backend.gather_masks(beam_masks, level, b_idx, s_idx)

            beam_scores, beam_masks = sel_scores, next_masks
            parents_per_level.append(self._to_cpu(b_idx))
            shifts_per_level.append(self._to_cpu(s_idx))
        if cfg.SHOW_PROGRESS:
            pbar.close()

        final_scores = self._to_cpu(beam_scores)
        if final_scores.size > 0:
            self.max_count = int(final_scores.max())
            best_local = np.where(final_scores == self.max_count)[0]
            self.results = int(best_local.size)

            n_levels = len(shifts_per_level)
            take = best_local if self.save_all_best else best_local[:1]
            for local_idx in take:
                path = []
                cur = int(local_idx)
                for lv in range(n_levels - 1, -1, -1):
                    path.append(int(shifts_per_level[lv][cur]))
                    cur = int(parents_per_level[lv][cur])
                path.reverse()
                self.best_paths.append(path)

        self.save_best_paths()
        elapsed = time.time() - start_time
        logger.info(
            "GPUビーム探索完了: 所要時間=%.2f秒, 探索ノード数=%d (%.2f nodes/s), backend=%s",
            elapsed, self.nodes_searched, self.nodes_searched / max(elapsed, 1e-6), self.backend.name,
        )
        return self

    # -- ヘルパ: backend非依存化のための小道具 --------------------------
    def _row_popcount(self, table_rows):
        """table_rows: (P, n_words) -> 各行の合計popcount (P,)"""
        if isinstance(self.backend, GPUBackend):
            cp = self.backend.cp
            x = table_rows.astype(cp.uint64)
            x = x - ((x >> cp.uint64(1)) & cp.uint64(0x5555555555555555))
            x = (x & cp.uint64(0x3333333333333333)) + ((x >> cp.uint64(2)) & cp.uint64(0x3333333333333333))
            x = (x + (x >> cp.uint64(4))) & cp.uint64(0x0F0F0F0F0F0F0F0F)
            x = (x * cp.uint64(0x0101010101010101)) >> cp.uint64(56)
            return x.astype(cp.int64).sum(axis=-1)
        else:
            return popcount64_numpy(table_rows).sum(axis=-1)

    def _nonzero(self, mask):
        xp = self.backend.cp if isinstance(self.backend, GPUBackend) else np
        return xp.where(mask)[0]

    def _topk_indices(self, scores, k):
        xp = self.backend.cp if isinstance(self.backend, GPUBackend) else np
        k = min(k, scores.shape[0])
        part = xp.argpartition(-scores, k - 1)[:k]
        order = xp.argsort(-scores[part])
        return part[order]

    def _select_topk(self, scores, masks, shifts, k):
        xp = self.backend.cp if isinstance(self.backend, GPUBackend) else np
        idx = self._topk_indices(scores, k)
        return scores[idx], masks[idx], shifts[idx]

    def _to_cpu(self, arr) -> np.ndarray:
        if isinstance(self.backend, GPUBackend):
            return self.backend.cp.asnumpy(arr)
        return np.asarray(arr)

    def save_best_paths(self) -> None:
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"max_count:{self.max_count}\n")
            f.write(f"results_count:{self.results}\n")
            for path in self.best_paths:
                f.write(f"{path}\n")
