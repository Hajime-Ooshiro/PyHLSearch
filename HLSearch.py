# HLSearch.py (改善版)
import logging
import logging.handlers
from pathlib import Path
from typing import List, Tuple
import time
import Config as cfg

# --- logging設定 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(cfg.LOG_FORMAT)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(cfg.CONSOLE_LOG_LEVEL)
_console_handler.setFormatter(_formatter)

_file_handler = logging.handlers.RotatingFileHandler(
    cfg.LOG_FILE,
    maxBytes=cfg.LOG_MAX_BYTES,
    backupCount=cfg.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_file_handler.setLevel(cfg.FILE_LOG_LEVEL)
_file_handler.setFormatter(_formatter)

if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)


def build_bit_tables(primes: List[int], cols: int) -> List[List[int]]:
    """
    各素数 p とそのシフト s (0 <= s < p) に対し、
    (idx - s) % p != 1 となる列を残すビットマスクを生成。
    """
    all_ones = (1 << cols) - 1
    tables = []
    for p in primes:
        p_shifts = []
        for s in range(p):
            # (idx - s) % p == 1 となるビットを 0 にする
            # idx = s + 1 + k * p
            drop_mask = 0
            start_idx = s + 1
            while start_idx <= 0:
                start_idx += p
            for idx in range(start_idx, cols + 1, p):
                drop_mask |= 1 << (idx - 1)
            p_shifts.append(all_ones ^ drop_mask)
        tables.append(p_shifts)
    return tables


class FastSearcher:
    def __init__(
        self,
        primes: List[int],
        depth: int,
        limit: int,
        target: int,
        max_depth: int,
        cols: int,
        output_file: Path,
        save_all_best: bool = False,
    ):
        self.primes = primes[:depth]
        self.depth = depth
        self.limit = limit
        self.target = target
        self.max_depth = max_depth
        self.cols = cols
        self.output_file = output_file
        self.save_all_best = save_all_best

        # 各深さ・シフトごとのビットマスク
        self.bit_tables = build_bit_tables(self.primes, cols)

        self.max_count = 0
        self.results = 0
        self.nodes_searched = 0
        self.shift_path: List[int] = []
        self.best_paths: List[List[int]] = []

    def run(self) -> "FastSearcher":
        start_time = time.time()
        initial_mask = (1 << self.cols) - 1

        first_p = self.primes[0]
        for s in range(first_p):
            self.shift_path.append(s)
            self._search(0, initial_mask & self.bit_tables[0][s])
            self.shift_path.pop()

        self.save_best_paths()

        elapsed = time.time() - start_time
        logger.info(
            "探索完了: 所要時間=%.2f秒, 探索ノード数=%d (%.2f nodes/s)",
            elapsed,
            self.nodes_searched,
            self.nodes_searched / max(elapsed, 1e-6),
        )
        return self

    def _search(self, level: int, current_mask: int) -> None:
        self.nodes_searched += 1
        count = current_mask.bit_count()

        # 枝刈り
        if count < self.limit or count < self.max_count:
            return

        # 最深部に到達
        if level + 1 >= self.depth:
            if self.depth == self.max_depth and count > self.target:
                return

            if count > self.max_count:
                self.max_count = count
                self.results = 1
                self.best_paths = [list(self.shift_path)]
                logger.info("New max_count=%d (path=%s)", self.max_count, self.shift_path)
            elif count == self.max_count:
                self.results += 1
                if self.save_all_best:
                    self.best_paths.append(list(self.shift_path))
            return

        # 子階層の展開 (Greedy / Best-First Ordering)
        next_level = level + 1
        next_p = self.primes[next_level]
        table_next = self.bit_tables[next_level]

        # 残存ビット数が多い順にソートして探索（高スコアを早期発見して枝刈り効率を最大化）
        candidates: List[Tuple[int, int, int]] = []
        for s in range(next_p):
            next_mask = current_mask & table_next[s]
            next_count = next_mask.bit_count()
            if next_count >= self.limit and next_count >= self.max_count:
                candidates.append((next_count, s, next_mask))

        # 残存ビット数の降順でソート
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, s, next_mask in candidates:
            self.shift_path.append(s)
            self._search(next_level, next_mask)
            self.shift_path.pop()

    def save_best_paths(self) -> None:
        """最良解をファイルに出力する"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"max_count:{self.max_count}\n")
            f.write(f"results_count:{self.results}\n")
            for path in self.best_paths:
                f.write(f"{path}\n")


if __name__ == "__main__":
    logger.info("HLSearch 開始 (log file: %s)", cfg.LOG_FILE)

    searcher = FastSearcher(
        primes=cfg.PRIMES,
        depth=cfg.DEPTH,
        limit=cfg.LIMIT,
        target=cfg.TARGET,
        max_depth=cfg.MAX_DEPTH,
        cols=cfg.COLS,
        output_file=cfg.SHIFT_PATH_FILE,
        save_all_best=False,
    )
    searcher.run()

    logger.info("最大値: %d", searcher.max_count)
    logger.info("該当件数: %d", searcher.results)
    logger.info("HLSearch 終了")