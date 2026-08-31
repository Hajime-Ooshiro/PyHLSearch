"""HLSearch_BitPack.py

NumPy を使わず、Python の int を bitset として扱う bit-packed 実装。

- `int` を使って各列の on/off をビットで表現する
- `bit_count()` を使って popcount を計算する
- `State` は元の `HLSearch.py` と同じ DFS ロジックを維持
- 依存: Python stdlib + tqdm

例:
    python HLSearch_BitPack.py --depth 8 --limit 400 --max-depth 249 --target 447
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm


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
    log_path = os.path.join(base_dir, "HLSearch_BitPack.log")
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


COLS: int = DEFAULT_CONFIG.cols
LIMIT: int = DEFAULT_CONFIG.limit
DEPTH: int = DEFAULT_CONFIG.depth
MAX_DEPTH: int = DEFAULT_CONFIG.max_depth
TARGET: int = DEFAULT_CONFIG.target
PROGRESS_MININTERVAL: float = DEFAULT_CONFIG.progress_mininterval
POSTFIX_UPDATE_INTERVAL: int = DEFAULT_CONFIG.postfix_update_interval
PRIMES: list[int] = DEFAULT_CONFIG.primes


def build_base_rows(primes: Sequence[int]) -> list[int]:
    """各素数 p ごとに、条件 `(idx % p) == 1` を満たす列だけ 1 となるビットマスクを生成する。"""
    rows: list[int] = []
    for p in primes:
        row = 0
        for idx in range(1, COLS + 1):
            if (idx % p) == 1:
                row |= 1 << (idx - 1)
        rows.append(row)
    return rows


def shift_int_bits(mask: int, k: int) -> int:
    """int のビット列を右へ k だけシフトして、欠けた分を 0 で埋める。"""
    if k <= 0:
        return mask
    if k >= COLS:
        return 0
    return (mask >> k) & ((1 << (COLS - k)) - 1)


def build_shift_table(primes: Sequence[int], cols: int = COLS) -> list[list[int]]:
    """各レベルのシフト候補をビット mask で事前生成する。"""
    base_rows = []
    for p in primes:
        row = 0
        for idx in range(1, cols + 1):
            if (idx % p) == 1:
                row |= 1 << (idx - 1)
        base_rows.append(row)

    shift_table: list[list[int]] = []
    for level, p in enumerate(primes):
        row = base_rows[level]
        level_masks: list[int] = []
        for k in range(p):
            shifted = shift_int_bits(row, k)
            complement = ((1 << cols) - 1) ^ shifted
            level_masks.append(complement)
        shift_table.append(level_masks)
    return shift_table


class State:
    """NumPy なし・bit-packed 版の探索状態。"""

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

    def __init__(self, config: SearchConfig | Sequence[int], shift_table: list[list[int]], limit: int | None = None, max_depth: int | None = None, target: int | None = None) -> None:
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
        self.shift_table: list[list[int]] = shift_table
        self.zero_mask: int = (1 << self.config.cols) - 1
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
        stack: list[tuple[int, int, Iterator[int]]] = [(0, self.zero_mask, iter(range(self.primes[0])))]

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
            count = node_mask.bit_count()

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
    parser = argparse.ArgumentParser(description="HLSearch bitpack: NumPy なしの素数シフト探索", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--depth", type=int, default=DEPTH, help="探索する階層数")
    parser.add_argument("-l", "--limit", type=int, default=LIMIT, help="枝刈り下限")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH, help="最大深さ")
    parser.add_argument("-t", "--target", type=int, default=TARGET, help="depth == max-depth のときの目標値")
    parser.add_argument("-p", "--primes-count", type=int, default=None, metavar="N", help="PRIMES の先頭 N 個だけ使用")
    parser.add_argument("--cols", type=int, default=COLS, help="列数）")
    parser.add_argument("--output", type=str, default=shift_path_file, help="最適シフトパスの出力先")
    parser.add_argument("--mininterval", type=float, default=PROGRESS_MININTERVAL, help="tqdm の最短更新間隔")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="ログレベル")
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

    logger.info("HLSearch_bitpack 開始 (log file: %s)", LOG_PATH)
    logger.info("設定: depth=%d limit=%d max_depth=%d target=%d primes_count=%d", depth, limit, max_depth, target, len(primes))

    shift_table = build_shift_table(primes[:depth], cols)
    state = State(config, shift_table)
    result_state = state.run(depth)

    logger.info("最大値: %d", result_state.max_count)
    logger.info("該当件数: %d", result_state.results)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"max_count:{result_state.max_count}\n")
        f.write(f"results:{result_state.results}\n")
        for shift in result_state.shifts:
            f.write(f"{shift}\n")

    logger.info("HLSearch_bitpack 終了")
