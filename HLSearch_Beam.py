"""HLSearch: 素数シフト探索プログラム。

このモジュールは、ハーディ・リトルウッドの第2予想に関する探索問題を
NumPy ベースの bitmask/shift table で高速化するための実装を提供する。

公開 API:
- generate_primes(limit): 指定上限までの素数を生成する
- SearchConfig: 探索設定をまとめた dataclass
- build_base_rows(primes): 素数ごとの基底行を生成する
- build_shift_table(primes): シフト候補テーブルを生成する
- State: 探索を実行する状態管理クラス
- parse_args(argv): CLI 引数を解析する

利用者はこのモジュールを直接 import して `State` を使うか、
`python HLSearch.py` として CLI から起動する。
"""

import argparse
import logging
import logging.handlers
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def generate_primes(limit: int) -> list[int]:
    """limit 以下の素数を昇順で返す。

    Args:
        limit: 上限値。2 以上を指定する。

    Returns:
        limit 以下の素数を昇順に並べたリスト。
    """
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start:limit + 1:p] = b"\x00" * (((limit - start) // p) + 1)
    return [n for n in range(2, limit + 1) if sieve[n]]


@dataclass(frozen=True)
class SearchConfig:
    """探索処理に必要な設定をまとめた構成体。

    Attributes:
        primes: 探索対象の素数リスト。デフォルトでは 1579 以下の素数を生成する。
        depth: 深さとして使う素数の数。
        limit: 枝刈りの下限値。
        max_depth: 深さの上限。
        target: `depth == max_depth` のときの打ち切り目標値.
        cols: 列数。
        progress_mininterval: tqdm の最短更新間隔。
        postfix_update_interval: postfix 更新の頻度。
        shift_path_file: 出力ファイルパス.
    """
    primes: Sequence[int] = field(default_factory=lambda: generate_primes(1579))
    depth: int = 8
    limit: int = 447
    max_depth: int = 249
    target: int = 447
    cols: int = 3159
    progress_mininterval: float = 1.0
    postfix_update_interval: int = 10000
    shift_path_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_path.txt")

DEFAULT_CONFIG = SearchConfig()
shift_path_file: str = DEFAULT_CONFIG.shift_path_file

# --- logging設定 ---
logger = logging.getLogger(__name__)

def setup_logging(base_dir: str | os.PathLike[str], console_level: str="INFO") -> str:
    """コンソールとファイルの両方にログを出力するよう設定する。"""
    log_path = os.path.join(base_dir, "HLSearch.log")

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 二重登録防止(再実行・再インポート対策)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"ログファイルを作成しました: {log_path}")
    return log_path


COLS: int = 3159
IDX: NDArray[np.int64] = np.arange(1, COLS + 1)  # range(1, 3160) 
LIMIT: int = 447

DEPTH: int = 8
MAX_DEPTH: int = 249
TARGET: int = 447
PROGRESS_MININTERVAL: float = 1.0  # tqdm進捗表示の最短更新間隔(秒)
POSTFIX_UPDATE_INTERVAL: int = 10000

# 2,3,5,7,11,13,...,1579 の素数リスト
PRIMES: list[int] = generate_primes(1579)
ROWS: int = len(PRIMES)

def shift_array(arr: NDArray[np.bool_], k: int) -> NDArray[np.bool_]:
    """
    配列を右にk個シフトする(numpy版)。先頭k要素は0埋めし、末尾のk要素は捨てる。
    """
    n = len(arr)
    if k <= 0:
        return arr.copy()
    if k >= n:
        return np.zeros(n, dtype=arr.dtype)
    result = np.empty(n, dtype=arr.dtype)
    result[:k] = 0
    result[k:] = arr[: n - k]
    return result
 

def build_base_rows(primes: Sequence[int]) -> NDArray[np.bool_]:
    """指定した素数リストから各階層の基底行を生成する。

    各要素は `bool((idx % p) == 1)` を保持し、探索では「0かどうか」だけを
    判定する。`bool` 型にすることで 1 要素あたりのメモリ使用量を抑え、
    シフトや AND 演算を高速化する。

    Args:
        primes: 基底行を作る素数一覧。

    Returns:
        shape=(len(primes), COLS) の bool 配列。
    """
    return np.array([(IDX % p == 1) for p in primes])


def build_shift_table(primes: Sequence[int]) -> list[NDArray[np.bool_]]:
    """各レベルごとのシフト候補テーブルを事前生成する。

    これにより探索時に毎回 `shift_array()` と `~` 演算を行わず、
    事前に補集合を計算済みの配列をそのまま使える。

    Args:
        primes: 使用する素数のリスト。

    Returns:
        `shift_table[level][shift]` が、level 段目におけるシフト値 `shift`
        に対応する補集合行を表す bool 配列。
    """
    base_rows = build_base_rows(primes)
    shift_table: list[NDArray[np.bool_]] = []
    for level, p in enumerate(primes):
        row = base_rows[level]
        shifted_complement = np.empty((p, COLS), dtype=bool)
        for k in range(p):
            shifted_complement[k] = ~shift_array(row, k)
        shift_table.append(shifted_complement)
    return shift_table


# --- CUDA / parallel backend support ---
BeamWidthSpec = int | list[int] | Callable[[int], int]


def build_bit_tables_packed(primes: Sequence[int], cols: int) -> tuple[list[NDArray[np.uint64]], int]:
    """各レベルの bitmask を packed uint64 配列へ変換し、CUDAで並列計算しやすくする。

    これにより `beam_masks[b] & table[s]` を 1 ワード単位で処理できる。
    """
    n_words = (cols + 63) // 64
    j = np.arange(cols, dtype=np.int64)
    tables: list[NDArray[np.uint64]] = []
    for p in primes:
        residues = j % p
        s_arr = np.arange(p, dtype=np.int64)[:, None]
        keep = residues[None, :] != s_arr

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


def popcount64_numpy(x: NDArray[np.uint64]) -> NDArray[np.int64]:
    """uint64 配列の各要素に対する popcount を計算する。"""
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return x.astype(np.int64)


class CPUBackend:
    """NumPy で並列化した候補スコア計算。"""

    name = "cpu(numpy)"

    def __init__(self, tables: list[NDArray[np.uint64]], n_words: int):
        self.tables = tables
        self.n_words = n_words

    def score_all(self, beam_masks: NDArray[np.uint64], level: int) -> NDArray[np.int64]:
        table = self.tables[level]
        anded = beam_masks[:, None, :] & table[None, :, :]
        counts = popcount64_numpy(anded).sum(axis=-1)
        return counts

    def gather_masks(self, beam_masks: NDArray[np.uint64], level: int, b_idx: NDArray[np.int64], s_idx: NDArray[np.int64]) -> NDArray[np.uint64]:
        table = self.tables[level]
        return beam_masks[b_idx] & table[s_idx]


_CUDA_KERNEL_SRC = r"""
extern "C" __global__
void score_kernel(
    const unsigned long long* __restrict__ beam_masks,
    const unsigned long long* __restrict__ table,
    long long* __restrict__ out_counts,
    int n_words, int P, int B)
{
    extern __shared__ unsigned long long shared_beam[];
    int b = blockIdx.x;
    if (b >= B) return;

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
    """CuPy/CUDA を使って候補スコア計算を GPU 上で並列化する。"""

    name = "cuda(cupy)"

    def __init__(self, tables_np: list[NDArray[np.uint64]], n_words: int, block_size: int = 128):
        import cupy as cp

        self.cp = cp
        self.n_words = n_words
        self.block_size = block_size
        self.tables = [cp.asarray(table) for table in tables_np]
        self._kernel = cp.RawKernel(_CUDA_KERNEL_SRC, "score_kernel")

    def score_all(self, beam_masks, level: int):
        cp = self.cp
        table = self.tables[level]
        B = beam_masks.shape[0]
        P = table.shape[0]
        out = cp.empty((B, P), dtype=cp.int64)

        block = self.block_size
        grid_y = max(1, (P + block - 1) // block)
        shared_bytes = self.n_words * 8

        self._kernel(
            (B, grid_y),
            (block,),
            (beam_masks, table, out, np.int32(self.n_words), np.int32(P), np.int32(B)),
            shared_mem=shared_bytes,
        )
        return out

    def gather_masks(self, beam_masks, level: int, b_idx, s_idx):
        table = self.tables[level]
        return beam_masks[b_idx] & table[s_idx]


class BeamSearcherGPU:
    """GPU/CPU を切り替え可能なビーム探索器。"""

    def __init__(
        self,
        primes: Sequence[int],
        depth: int,
        limit: int,
        target: int,
        max_depth: int,
        cols: int,
        output_file: str | os.PathLike[str],
        beam_width: int | list[int] | Callable[[int], int] = 100,
        save_all_best: bool = False,
        use_gpu: bool = True,
    ):
        self.primes = list(primes)[:depth]
        self.depth = depth
        self.limit = limit
        self.target = target
        self.max_depth = max_depth
        self.cols = cols
        self.output_file = Path(output_file)
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
        self.best_paths: list[list[int]] = []

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

    def _row_popcount(self, table_rows):
        if isinstance(self.backend, GPUBackend):
            cp = self.backend.cp
            x = table_rows.astype(cp.uint64)
            x = x - ((x >> cp.uint64(1)) & cp.uint64(0x5555555555555555))
            x = (x & cp.uint64(0x3333333333333333)) + ((x >> cp.uint64(2)) & cp.uint64(0x3333333333333333))
            x = (x + (x >> cp.uint64(4))) & cp.uint64(0x0F0F0F0F0F0F0F0F)
            x = (x * cp.uint64(0x0101010101010101)) >> cp.uint64(56)
            return x.astype(cp.int64).sum(axis=-1)
        return popcount64_numpy(table_rows).sum(axis=-1)

    def _nonzero(self, mask):
        xp = self.backend.cp if isinstance(self.backend, GPUBackend) else np
        return xp.where(mask)[0]

    def _topk_indices(self, scores, k):
        xp = self.backend.cp if isinstance(self.backend, GPUBackend) else np
        k = min(k, scores.shape[0])
        if k <= 0:
            return xp.empty(0, dtype=np.int64)
        part = xp.argpartition(-scores, k - 1)[:k]
        order = xp.argsort(-scores[part])
        return part[order]

    def _select_topk(self, scores, masks, shifts, k):
        xp = self.backend.cp if isinstance(self.backend, GPUBackend) else np
        idx = self._topk_indices(scores, k)
        return scores[idx], masks[idx], shifts[idx]

    def _to_cpu(self, arr) -> NDArray[np.int64] | NDArray[np.uint64]:
        if isinstance(self.backend, GPUBackend):
            return self.backend.cp.asnumpy(arr)
        return np.asarray(arr)

    def run(self) -> "BeamSearcherGPU":
        import time

        xp = self.xp
        start_time = time.time()

        first_p = self.primes[0]
        table0 = self.backend.tables[0]
        counts0 = self._row_popcount(table0)

        self.nodes_searched += first_p
        valid = counts0 >= self.limit
        idx_valid = xp.where(valid)[0]

        scores = counts0[idx_valid]
        masks = table0[idx_valid]
        shifts = idx_valid

        beam_scores, beam_masks, beam_shifts = self._select_topk(scores, masks, shifts, self._resolve_beam_width(0))
        parents_per_level: list[NDArray[np.int64]] = [np.full(len(beam_shifts), -1, dtype=np.int64)]
        shifts_per_level: list[NDArray[np.int64]] = [self._to_cpu(beam_shifts)]

        for level in range(1, self.depth):
            B = beam_masks.shape[0]
            P = self.primes[level]
            is_last_level = (level + 1 >= self.depth)

            all_counts = self.backend.score_all(beam_masks, level)
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

        final_scores = self._to_cpu(beam_scores)
        if final_scores.size > 0:
            self.max_count = int(final_scores.max())
            best_local = np.where(final_scores == self.max_count)[0]
            self.results = int(best_local.size)

            n_levels = len(shifts_per_level)
            take = best_local if self.save_all_best else best_local[:1]
            for local_idx in take:
                path: list[int] = []
                cur = int(local_idx)
                for lv in range(n_levels - 1, -1, -1):
                    path.append(int(shifts_per_level[lv][cur]))
                    cur = int(parents_per_level[lv][cur])
                path.reverse()
                self.best_paths.append(path)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as f:
            f.write(f"max_count:{self.max_count}\n")
            f.write(f"results_count:{self.results}\n")
            for path in self.best_paths:
                f.write(f"{path}\n")

        elapsed = time.time() - start_time
        logger.info(
            "GPUビーム探索完了: 所要時間=%.2f秒, 探索ノード数=%d (%.2f nodes/s), backend=%s",
            elapsed,
            self.nodes_searched,
            self.nodes_searched / max(elapsed, 1e-6),
            self.backend.name,
        )
        return self


class State:
    """探索処理の状態を保持し、反復 DFS を実行する。

    This class owns the current search path (`key`), the active zero-mask,
    the pruning thresholds, and the best result state seen so far.

    Public API:
    - `search(depth)`: 指定深さまで DFS を実行する
    - `run(depth=None)`: 既定設定を使って探索を実行し、self を返す
    - `max_count`, `results`, `shifts`: 最良結果の集計
    """

    __slots__ = (
        "config",
        "key",
        "primes",
        "shift_table",
        "zero_mask",
        "limit",
        "max_depth",
        "target",
        "max_count",
        "shifts",
        "results",
        "start_time",
        "node_count",
        "pbar",
    )

    def __init__(self, config: SearchConfig | Sequence[int], shift_table: list[NDArray[np.bool_]], limit: int | None = None, max_depth: int | None = None, target: int | None = None) -> None:
        if isinstance(config, SearchConfig):
            self.config = config
            primes = config.primes
            self.limit = config.limit if limit is None else limit
            self.max_depth = config.max_depth if max_depth is None else max_depth
            self.target = config.target if target is None else target
        else:
            self.config = SearchConfig(primes=config, depth=len(config), limit=LIMIT if limit is None else limit, max_depth=MAX_DEPTH if max_depth is None else max_depth, target=TARGET if target is None else target, cols=COLS)
            primes = config
            self.limit = LIMIT if limit is None else limit
            self.max_depth = MAX_DEPTH if max_depth is None else max_depth
            self.target = TARGET if target is None else target

        self.key: list[int] = []
        self.primes: Sequence[int] = primes
        self.shift_table: list[NDArray[np.bool_]] = shift_table
        self.zero_mask: NDArray[np.bool_] = np.ones(COLS, dtype=bool)
        self.max_count: int = 0
        self.shifts: list[list[int]] = []
        self.results: int = 0
        self.start_time: float = time.time()
        self.node_count: int = 0
        self.pbar = tqdm(
            desc="search",
            unit="node",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=self.config.progress_mininterval if isinstance(self.config, SearchConfig) else PROGRESS_MININTERVAL,
        )

    def report_progress(self, force: bool = False) -> None:
        """
        tqdmの進捗バーを更新する(ログファイルには出さない)。

        呼び出しが多い(再帰の全ノードで呼ばれる)ため、実際の画面再描画は
        tqdm側が mininterval 秒に一度だけに間引いてくれる。force=True の
        ときは set_postfix に refresh=True を渡し、間引かずに必ず再描画する
        (探索開始・終了時など)。
        """
        self.pbar.update(1)
        if self.node_count % self.config.postfix_update_interval == 0:
            self.pbar.set_postfix(
                best=self.max_count,
                hits=self.results,
                depth=len(self.key),
                key=list(self.key),
                refresh=force,
            )

    def search(self, depth: int) -> None:
        """
        各階層のシフト値を探索する(反復版・スタックによる明示的DFS)。

        元の実装は「1階層シフトを決める→自分自身を再帰呼び出しして
        次の階層を決める」という再帰関数だったが、depth(ひいては
        再帰の深さ)が大きくなると Python の再帰上限(sys.setrecursionlimit)
        や関数呼び出しオーバーヘッドが問題になりうる。
        ここでは再帰呼び出しの代わりに、階層ごとの「ループの途中状態」を
        自前のスタックに積んで管理することで、同じ探索順序・同じ結果を
        非再帰(反復)で実現する。

        スタックの各要素は (level, base_mask, iterator) の3つ組:
            level      : このループで値を決める階層(0-indexed)。
                         元の再帰版での level = len(key) - 1 に対応。
            base_mask  : この階層のどの枝を試す場合でも共通して使う
                         「親までの zero_mask」。元の再帰版での
                         prev_mask(= self.zero_mask の呼び出し前の値)
                         に対応する。
            iterator   : range(primes[level]) の残り候補値を返す
                         イテレータ。元の re-entrant な `for i in range(...)`
                         ループの「途中状態」をこれで表現する。

        1つの節点(= key の1要素)を「探索し尽くして親に戻る」タイミングは、
        自分の子階層のイテレータが尽きた(StopIteration)瞬間として検出し、
        そこで元の再帰版の「self.zero_mask = prev_mask; return」に相当する
        後始末(zero_maskの復元・keyのpop)を行う。

        Parameters
        ----------
        depth : int
            探索する階層数(= 使用する素数の個数)。可変。
        """
        key = self.key
        stack: list[tuple[int, NDArray[np.bool_], Iterator[int]]] = [
            (0, self.zero_mask, iter(range(self.primes[0])))
        ]

        while stack:
            level, base_mask, it = stack[-1]
            try:
                i = next(it)
            except StopIteration:
                # この階層で試せる値を使い切った → 親の階層へbacktrack
                finished_base_mask = stack.pop()[1]
                if stack:
                    key.pop()
                    self.zero_mask = stack[-1][1]
                else:
                    self.zero_mask = finished_base_mask  # 最上位まで戻り切った
                continue

            key.append(i)
            self.node_count += 1
            self.report_progress()

            row_complement = self.shift_table[level][i]  # ~row_nonzero(NOT演算済み、事前作成済み)
            node_mask = base_mask & row_complement
            count = int(np.count_nonzero(node_mask))
            # logger.debug("depth=%d key=%s count=%s", level + 1, key, count)

            if count < max(self.limit, self.max_count):
                # logger.debug(("break", list(key), count))
                key.pop()  # この枝は打ち切り(子孫を探索しない)、次のiへ
                continue

            if level + 1 >= depth:
                if not (depth == self.max_depth and count > self.target):
                    if count > self.max_count:
                        self.max_count = count
                        self.results = 1
                        self.shifts.clear()
                        self.shifts.append(list(key))
                        self.pbar.write(f"done key={list(key)} count={count}")
                        # logger.debug(("done", list(key), count))
                    elif count == self.max_count:
                        self.results += 1
                        self.shifts.append(list(key))
                        # logger.debug(("done", list(key), count))

                key.pop()  # 最深階層に到達、次のiへ
                continue

            # さらに深く探索: 子階層のループをスタックに積んで先に進む
            self.zero_mask = node_mask
            next_p = self.primes[level + 1]
            stack.append((level + 1, node_mask, iter(range(next_p))))

    def run(self, depth: int | None = None) -> "State":
        """primes[:depth] を使って深さ depth までの探索を実行するエントリポイント"""
        depth_to_use = self.config.depth if depth is None else depth
        if depth_to_use > len(self.primes):
            raise ValueError(f"depth={depth_to_use} が使用可能な素数の個数({len(self.primes)})を超えています")

        try:
            self.search(depth_to_use)
            self.report_progress(force=True)
        finally:
            self.pbar.close()

        return self


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI 引数を解釈して探索設定を返す。

    Args:
        argv: 引数リスト。None の場合は `sys.argv[1:]` を使う。

    Returns:
        argparse.Namespace 形式の設定。
    """
    parser = argparse.ArgumentParser(
        description="HLSearch: 素数シフト探索プログラム",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--depth", type=int, default=DEPTH,
                        help="探索する階層数(使用する素数の個数)。primesの長さ以下である必要がある。")
    parser.add_argument("-l", "--limit", type=int, default=LIMIT,
                        help="打ち切りに使うcountの下限値。これ未満の枝は探索しない。")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                        help="depthがこの値と一致するとき、--targetによる追加打ち切りを有効にする。")
    parser.add_argument("-t", "--target", type=int, default=TARGET,
                        help="depth == max-depth のとき、countがこの値を超えたら結果を採用せず打ち切る。")
    parser.add_argument("-p", "--primes-count", type=int, default=None, metavar="N",
                        help="PRIMESの先頭N個だけを使う(未指定なら全て使用)。")
    parser.add_argument("--cols", type=int, default=COLS,
                        help="列数(=探索対象の長さ)。")
    parser.add_argument("--output", type=str, default=shift_path_file,
                        help="最適シフトパスの出力先ファイル。")
    parser.add_argument("--mininterval", type=float, default=PROGRESS_MININTERVAL,
                        help="tqdm進捗表示の最短更新間隔(秒)。")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="コンソールに出すログレベル(ログファイルは常にDEBUG)。")
    parser.add_argument("--gpu", action="store_true",
                        help="CUDA/CuPy バックエンドを優先して使う(利用できない場合は自動で CPU にフォールバック)。")
    return parser.parse_args(argv)


if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0]).lower()
    argv = sys.argv[1:] if script_name not in {"pytest", "py.test"} else []
    args = parse_args(argv)

    base = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = setup_logging(base)

    depth = args.depth
    limit = args.limit
    max_depth = args.max_depth
    target = args.target
    cols = args.cols
    primes = PRIMES if args.primes_count is None else PRIMES[: args.primes_count]
    output_path = args.output

    PROGRESS_MININTERVAL = args.mininterval

    if depth > len(primes):
        raise ValueError(f"depth={depth} が使用可能な素数の個数({len(primes)})を超えています")

    config = SearchConfig(
        primes=primes,
        depth=depth,
        limit=limit,
        max_depth=max_depth,
        target=target,
        cols=cols,
        progress_mininterval=args.mininterval,
        postfix_update_interval=POSTFIX_UPDATE_INTERVAL,
        shift_path_file=output_path,
    )

    logger.info("HLSearch 開始 (log file: %s)", LOG_PATH)
    logger.info("設定: depth=%d limit=%d max_depth=%d target=%d primes_count=%d", depth, limit, max_depth, target, len(primes))

    if args.gpu:
        result_state = BeamSearcherGPU(
            primes=primes,
            depth=depth,
            limit=limit,
            target=target,
            max_depth=max_depth,
            cols=cols,
            output_file=output_path,
            beam_width=100,
            use_gpu=True,
        ).run()
    else:
        shift_table = build_shift_table(primes[:depth])
        state = State(config, shift_table)
        result_state = state.run(depth)

    logger.info("最大値: %d", result_state.max_count)
    logger.info("該当件数: %d", result_state.results)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.gpu:
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"max_count:{result_state.max_count}\n")
            f.write(f"results:{result_state.results}\n")
            for shift in result_state.shifts:
                f.write(f"{shift}\n")

    logger.info("HLSearch 終了")