# HLSearch.py
import logging
import logging.handlers
from typing import List
import time
import Config as cfg

# --- Debug設定
DEBUG = cfg.DEBUG


# --- logging設定(設定値はすべて config.py に集約) ---
LOG_FILE = cfg.LOG_FILE
LOG_MAX_BYTES = cfg.LOG_MAX_BYTES
LOG_BACKUP_COUNT = cfg.LOG_BACKUP_COUNT
CONSOLE_LOG_LEVEL = cfg.CONSOLE_LOG_LEVEL
FILE_LOG_LEVEL = cfg.FILE_LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(cfg.LOG_FORMAT)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(CONSOLE_LOG_LEVEL)
_console_handler.setFormatter(_formatter)

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_file_handler.setLevel(FILE_LOG_LEVEL)
_file_handler.setFormatter(_formatter)

if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)

primes: List[int] = cfg.PRIMES
depth: int = cfg.DEPTH
limit: int = cfg.LIMIT
target: int = cfg.TARGET
max_depth: int =cfg.MAX_DEPTH
cols: int = cfg.COLS
shift_path_file: str = cfg.SHIFT_PATH_FILE


def build_bit_tables(primes: List[int], cols: int) -> List[List[int]]:
    """
    各素数 p とそのシフト s (0 <= s < p) に対して、
    各列 idx (1 <= idx <= cols) で (idx - s) % p == 1 となるビットを 0、
    それ以外を 1 としたビットマスク (AND演算で生き残る列) を事前作成する。
    """
    tables = []
    for p in primes:
        p_shifts = []
        for s in range(p):
            # 1-indexed の列 idx について、(idx - s - 1) % p == 0 が非ゼロとなる
            mask = 0
            for idx in range(1, cols + 1):
                # (idx - s) % p == 1 のとき該当素数の倍数位置（=0でなくなる）
                if (idx - s) % p != 1:
                    mask |= 1 << (idx - 1)
            p_shifts.append(mask)
        tables.append(p_shifts)
    return tables

class FastSearcher:
    def __init__(self, primes: List[int], depth: int, limit: int, target: int, max_depth: int, cols: int):
        self.primes = primes[:depth]
        self.depth = depth
        self.limit = limit
        self.target = target
        self.max_depth = max_depth
        self.cols = cols

        # 各深さ・シフトごとのビットマスク
        self.bit_tables = build_bit_tables(self.primes, cols)

        self.max_count = 0
        self.results = 0
        self.nodes_searched = 0
        self.shift_path: List[int] = []
        self.best_paths: List[List[int]] = []

    def run(self):
        start_time = time.time()
        # 初期マスク: 全 COLS ビットが 1 (すべて0候補)
        initial_mask = (1 << self.cols) - 1
        
        first_p = self.primes[0]
        for s in range(first_p):
            self.shift_path.append(s)
            self._search(0, initial_mask & self.bit_tables[0][s])
            self.shift_path.pop()
        self.append_shift_path_file()

        elapsed = time.time() - start_time
        logger.info("探索完了: 所要時間=%.2f秒, 探索ノード数=%d (%.2f nodes/s)",
                    elapsed, self.nodes_searched, self.nodes_searched / max(elapsed, 1e-6))
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
                # self.best_paths.append(list(self.shift_path))
            return

        # 子階層の探索
        next_level = level + 1
        next_p = self.primes[next_level]
        table_next = self.bit_tables[next_level]

        for s in range(next_p):
            self.shift_path.append(s)
            self._search(next_level, current_mask & table_next[s])
            self.shift_path.pop()

    def append_shift_path_file(self) -> None:
        """shift_path.txt に shift_path を1行追加する"""
        with open(shift_path_file, "w", encoding="utf-8") as f:
            f.write(f"max_count:{self.max_count}\n")
            for path in self.best_paths:
                f.write(f"{path}\n")


if __name__ == "__main__":
    logger.info("HLSearch 開始 (log file: %s)", LOG_FILE)

    DEPTH = cfg.DEPTH  # 使用する素数の個数(可変)。元コードの calc2->calc3->calc5->calc7 相当
    searcher = FastSearcher(primes, depth, limit, target, max_depth, cols)
    searcher.run()

    logger.info("最大値: %d", searcher.max_count)
    logger.info("該当件数: %d", searcher.results)

    logger.info("HLSearch 終了")