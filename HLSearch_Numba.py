"""HLSearch_Numba.py

Numba ベースの高速化版。既存の `HLSearch.py` の探索ロジックを維持しつつ、
ビットマスク計算のコア部分を Numba JIT に載せる。

対応機能:
- CPU: `njit(parallel=True)` による高速化
- CUDA: numba.cuda が利用可能なら GPU カーネルを選択可能
- フォールバック: Numba 未導入時は通常の NumPy 実装へ自動フォールバック

実行例:
    python HLSearch_Numba.py --depth 8 --limit 400 --max-depth 249 --target 447 --numba
"""

from __future__ import annotations

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

try:
    from numba import cuda, njit, prange
    HAS_NUMBA = True
except Exception:  # pragma: no cover - 環境依存
    HAS_NUMBA = False
    cuda = None

    def njit(*args, **kwargs):
        def deco(func):
            return func
        return deco

    def prange(x):
        return range(x)


def generate_primes(limit: int) -> list[int]:
    """limit 以下の素数を昇順で返す。"""
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
    primes: Sequence[int] = field(default_factory=lambda: generate_primes(1579))
    depth: int = 8
    limit: int = 447
    max_depth: int = 249
    target: int = 447
    cols: int = 3159
    progress_mininterval: float = 1.0
    postfix_update_interval: int = 10000
    shift_path_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_path.txt")

    def __post_init__(self) -> None:
        """設定値の整合性を早期に検証する(実行時ではなく構築時に失敗させる)。"""
        if self.cols <= 0:
            raise ValueError(f"cols は正の整数である必要があります: cols={self.cols}")
        if self.depth < 0:
            raise ValueError(f"depth は0以上である必要があります: depth={self.depth}")
        if self.depth > len(self.primes):
            raise ValueError(
                f"depth={self.depth} が primes の要素数({len(self.primes)})を超えています"
            )
        if self.limit < 0:
            raise ValueError(f"limit は0以上である必要があります: limit={self.limit}")
        if self.postfix_update_interval <= 0:
            raise ValueError(
                f"postfix_update_interval は正の整数である必要があります: "
                f"postfix_update_interval={self.postfix_update_interval}"
            )


DEFAULT_CONFIG = SearchConfig()
shift_path_file: str = DEFAULT_CONFIG.shift_path_file
logger = logging.getLogger(__name__)


def setup_logging(base_dir: str | os.PathLike[str], console_level: str = "INFO") -> str:
    log_path = os.path.join(base_dir, "HLSearch_Numba.log")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("ログファイルを作成しました: %s", log_path)
    return log_path


COLS: int = 3159
IDX: NDArray[np.int64] = np.arange(1, COLS + 1)
LIMIT: int = 447
DEPTH: int = 8
MAX_DEPTH: int = 249
TARGET: int = 447
PROGRESS_MININTERVAL: float = 1.0
POSTFIX_UPDATE_INTERVAL: int = 10000
PRIMES: list[int] = generate_primes(1579)


def shift_array(arr: NDArray[np.bool_], k: int) -> NDArray[np.bool_]:
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
    return np.array([(IDX % p == 1) for p in primes])


def build_shift_table(primes: Sequence[int]) -> list[NDArray[np.bool_]]:
    base_rows = build_base_rows(primes)
    shift_table: list[NDArray[np.bool_]] = []
    for level, p in enumerate(primes):
        row = base_rows[level]
        shifted_complement = np.empty((p, COLS), dtype=bool)
        for k in range(p):
            shifted_complement[k] = ~shift_array(row, k)
        shift_table.append(shifted_complement)
    return shift_table


BeamWidthSpec = int | list[int] | Callable[[int], int]


def build_bit_tables_packed(primes: Sequence[int], cols: int) -> tuple[list[NDArray[np.uint64]], int]:
    """packed uint64 mask table: CUDA/Numba のループに適した表現"""
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


@njit(cache=True, fastmath=True)
def _popcount_u64(v: np.uint64) -> int:
    x = int(v)
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x * 0x0101010101010101) >> 56
    return int(x)


@njit(cache=True, fastmath=True, parallel=True)
def score_all_numba(beam_masks: NDArray[np.uint64], table: NDArray[np.uint64]) -> NDArray[np.int64]:
    B = beam_masks.shape[0]
    P = table.shape[0]
    n_words = beam_masks.shape[1]
    out = np.empty((B, P), dtype=np.int64)
    for b in prange(B):
        for s in range(P):
            cnt = 0
            for w in range(n_words):
                v = int(beam_masks[b, w] & table[s, w])
                cnt += _popcount_u64(np.uint64(v))
            out[b, s] = cnt
    return out


@njit(cache=True, fastmath=True)
def row_popcount_numba(table_rows: NDArray[np.uint64]) -> NDArray[np.int64]:
    P = table_rows.shape[0]
    n_words = table_rows.shape[1]
    out = np.empty(P, dtype=np.int64)
    for s in range(P):
        cnt = 0
        for w in range(n_words):
            v = int(table_rows[s, w])
            cnt += _popcount_u64(np.uint64(v))
        out[s] = cnt
    return out


class NumbaCPUBackend:
    name = "numba(cpu)"

    def __init__(self, tables: list[NDArray[np.uint64]], n_words: int):
        self.tables = tables
        self.n_words = n_words

    def score_all(self, beam_masks: NDArray[np.uint64], level: int) -> NDArray[np.int64]:
        return score_all_numba(beam_masks, self.tables[level])

    def gather_masks(self, beam_masks: NDArray[np.uint64], level: int, b_idx: NDArray[np.int64], s_idx: NDArray[np.int64]) -> NDArray[np.uint64]:
        table = self.tables[level]
        return beam_masks[b_idx] & table[s_idx]


if HAS_NUMBA and cuda is not None:
    @cuda.jit
    def cuda_score_kernel(beam_masks, table, out_counts, n_words, P, B):
        b = cuda.blockIdx.x
        if b >= B:
            return
        shared_beam = cuda.shared.array(shape=(1024,), dtype=np.uint64)
        for w in range(cuda.threadIdx.x, n_words, cuda.blockDim.x):
            shared_beam[w] = beam_masks[b, w]
        cuda.syncthreads()
        for s in range(cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x, P, cuda.gridDim.y * cuda.blockDim.x):
            cnt = 0
            for w in range(n_words):
                v = int(shared_beam[w] & table[s, w])
                x = v
                x = x - ((x >> 1) & 0x5555555555555555)
                x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
                x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
                x = (x * 0x0101010101010101) >> 56
                cnt += x
            out_counts[b, s] = cnt


    class NumbaCUDABackend:
        name = "numba(cuda)"

        def __init__(self, tables_np: list[NDArray[np.uint64]], n_words: int, block_size: int = 128):
            self.cp = cuda
            self.n_words = n_words
            self.block_size = block_size
            self.tables = [cuda.to_device(t) for t in tables_np]

        def score_all(self, beam_masks, level: int):
            table = self.tables[level]
            B = beam_masks.shape[0]
            P = table.shape[0]
            out = cuda.device_array((B, P), dtype=np.int64)
            grid_y = max(1, (P + self.block_size - 1) // self.block_size)
            cuda_score_kernel[(B, grid_y), (self.block_size,)](beam_masks, table, out, np.int32(self.n_words), np.int32(P), np.int32(B))
            return out

        def gather_masks(self, beam_masks, level: int, b_idx, s_idx):
            table = self.tables[level]
            return beam_masks[b_idx] & table[s_idx]
else:
    NumbaCUDABackend = None


class BeamSearcherNumba:
    """Numba CPU/CUDA を使うビーム探索器。"""

    def __init__(
        self,
        primes: Sequence[int],
        depth: int,
        limit: int,
        target: int,
        max_depth: int,
        cols: int,
        output_file: str | os.PathLike[str],
        beam_width: BeamWidthSpec = 100,
        save_all_best: bool = False,
        use_cuda: bool = False,
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
        if use_cuda and NumbaCUDABackend is not None:
            try:
                self.backend = NumbaCUDABackend(tables_np, n_words)
            except Exception:
                logger.warning("Numba CUDA を初期化できないため CPU バックエンドにフォールバックします")

        if self.backend is None:
            self.backend = NumbaCPUBackend(tables_np, n_words)

        self.xp = np if not hasattr(self.backend, "cp") else np
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
        if isinstance(self.backend, NumbaCPUBackend):
            return row_popcount_numba(table_rows)
        if hasattr(self.backend, "cp") and self.backend.cp is cuda:
            return row_popcount_numba(cuda.as_numpy(table_rows))
        return row_popcount_numba(table_rows)

    def _nonzero(self, mask):
        return np.where(mask)[0]

    def _topk_indices(self, scores, k):
        k = min(k, scores.shape[0])
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        part = np.argpartition(-scores, k - 1)[:k]
        order = np.argsort(-scores[part])
        return part[order]

    def _select_topk(self, scores, masks, shifts, k):
        idx = self._topk_indices(scores, k)
        return scores[idx], masks[idx], shifts[idx]

    def _to_cpu(self, arr):
        if hasattr(self.backend, "cp") and self.backend.cp is cuda:
            return cuda.as_numpy(arr)
        return np.asarray(arr)

    def run(self) -> "BeamSearcherNumba":
        import time

        start_time = time.time()
        first_p = self.primes[0]
        table0 = self.backend.tables[0]
        counts0 = self._row_popcount(table0)

        self.nodes_searched += first_p
        valid = counts0 >= self.limit
        idx_valid = np.where(valid)[0]

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

            flat = np.asarray(all_counts).reshape(-1)
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
            "Numbaビーム探索完了: 所要時間=%.2f秒, 探索ノード数=%d (%.2f nodes/s), backend=%s",
            elapsed,
            self.nodes_searched,
            self.nodes_searched / max(elapsed, 1e-6),
            self.backend.name,
        )
        return self


class State:
    """既存の HLSearch.State と同じ挙動を維持する通常版。"""

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
        self.pbar = tqdm(desc="search", unit="node", unit_scale=True, dynamic_ncols=True, mininterval=self.config.progress_mininterval)

    def report_progress(self, force: bool = False) -> None:
        if self.node_count % self.config.postfix_update_interval == 0:
            self.pbar.update(1)
            self.pbar.set_postfix(best=self.max_count, hits=self.results, depth=len(self.key), key=list(self.key), refresh=force)

    def search(self, depth: int) -> None:
        key = self.key
        stack: list[tuple[int, NDArray[np.bool_], Iterator[int]]] = [(0, self.zero_mask, iter(range(self.primes[0])))]

        while stack:
            level, base_mask, it = stack[-1]
            try:
                i = next(it)
            except StopIteration:
                finished_base_mask = stack.pop()[1]
                if stack:
                    key.pop()
                    self.zero_mask = stack[-1][1]
                else:
                    self.zero_mask = finished_base_mask
                continue

            key.append(i)
            self.node_count += 1
            self.report_progress()

            row_complement = self.shift_table[level][i]
            node_mask = base_mask & row_complement
            count = int(np.count_nonzero(node_mask))

            if count < max(self.limit, self.max_count):
                key.pop()
                continue

            if level + 1 >= depth:
                if not (depth == self.max_depth and count > self.target):
                    if count > self.max_count:
                        self.max_count = count
                        self.results = 1
                        self.shifts.clear()
                        self.shifts.append(list(key))
                    elif count == self.max_count:
                        self.results += 1
                        self.shifts.append(list(key))
                key.pop()
                continue

            self.zero_mask = node_mask
            next_p = self.primes[level + 1]
            stack.append((level + 1, node_mask, iter(range(next_p))))

    def run(self, depth: int | None = None) -> "State":
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
    parser = argparse.ArgumentParser(description="HLSearch Numba: 素数シフト探索プログラム", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--depth", type=int, default=DEPTH, help="探索する階層数")
    parser.add_argument("-l", "--limit", type=int, default=LIMIT, help="枝刈り下限")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH, help="最大深さ")
    parser.add_argument("-t", "--target", type=int, default=TARGET, help="depth == max-depth のときの目標値")
    parser.add_argument("-p", "--primes-count", type=int, default=None, metavar="N", help="PRIMES の先頭 N 個だけ使用")
    parser.add_argument("--cols", type=int, default=COLS, help="列数")
    parser.add_argument("--output", type=str, default=shift_path_file, help="結果出力先")
    parser.add_argument("--mininterval", type=float, default=PROGRESS_MININTERVAL, help="tqdm の最短更新間隔")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="コンソールログレベル")
    parser.add_argument("--numba", action="store_true", help="Numba 版を使う（無い場合は通常版へフォールバック）")
    parser.add_argument("--cuda", action="store_true", help="Numba CUDA を優先して使う（不可なら CPU へフォールバック）")
    return parser.parse_args(argv)


if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0]).lower()
    argv = sys.argv[1:] if script_name not in {"pytest", "py.test"} else []
    args = parse_args(argv)

    base = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = setup_logging(base, args.log_level)

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

    logger.info("HLSearch_Numba 開始 (log file: %s)", LOG_PATH)
    logger.info("設定: depth=%d limit=%d max_depth=%d target=%d primes_count=%d", depth, limit, max_depth, target, len(primes))

    if args.numba or args.cuda:
        searcher = BeamSearcherNumba(
            primes=primes,
            depth=depth,
            limit=limit,
            target=target,
            max_depth=max_depth,
            cols=cols,
            output_file=output_path,
            beam_width=100,
            use_cuda=args.cuda,
        )
        result_state = searcher.run()
    else:
        shift_table = build_shift_table(primes[:depth])
        state = State(config, shift_table)
        result_state = state.run(depth)

    logger.info("最大値: %d", result_state.max_count)
    logger.info("該当件数: %d", result_state.results)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not (args.numba or args.cuda):
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"max_count:{result_state.max_count}\n")
            f.write(f"results:{result_state.results}\n")
            for shift in result_state.shifts:
                f.write(f"{shift}\n")

    logger.info("HLSearch_Numba 終了")
