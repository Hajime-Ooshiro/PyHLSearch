# HLSearch.py（厳密解法版）
#
# 探索アルゴリズム:
#   --exact : FastSearcher (DFS+分枝限定法。厳密解。CPU(bigint)のみ)
#
import logging
import logging.handlers
from pathlib import Path
from typing import List, Tuple, Optional
import time
from tqdm import tqdm
import Config as cfg

from cli_common import build_parser, ResolvedConfig, setup_logging

# モジュールレベルではロガーのハンドルだけ用意し、実際の設定(ハンドラ登録)は
# __main__ 内で CLI引数 / Config.py の値が確定してから行う。
logger = logging.getLogger(__name__)


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
    """深さ優先 + 分枝限定法（厳密解法）。
    有効な（limit/max_countを満たす）分岐は全て探索するため、
    枝刈りが効けば厳密な最良解が得られる。
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
        if cfg.SHOW_PROGRESS:
            self.pbar = tqdm(
                total=self.depth,
                desc="Searching...",
                unit="task",
                dynamic_ncols=True,
                )
        for s in range(first_p):
            self.shift_path.append(s)
            self._search(0, initial_mask & self.bit_tables[0][s])
            self.shift_path.pop()
        if cfg.SHOW_PROGRESS:
            self.pbar.close()

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
        if cfg.SHOW_PROGRESS:
            self.pbar.update(1)

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


def main() -> None:
    parser = build_parser(
        description="HLSearch: 厳密解法によるシフト探索",
        include_gpu_flag=False,
    )
    args = parser.parse_args()
    rc = ResolvedConfig(args, cfg)
    rc.apply_show_progress_override(cfg)

    setup_logging(
        logger_name=__name__,
        log_file=rc.log_file,
        console_level=rc.log_level,
        file_level=getattr(cfg, "FILE_LOG_LEVEL", "DEBUG"),
        log_format=getattr(cfg, "LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
        max_bytes=getattr(cfg, "LOG_MAX_BYTES", 10 * 1024 * 1024),
        backup_count=getattr(cfg, "LOG_BACKUP_COUNT", 3),
    )

    logger.info("HLSearch 開始 (log file: %s)", rc.log_file)
    logger.debug(
        "実行設定: depth=%s limit=%s target=%s max_depth=%s cols=%s output_file=%s save_all_best=%s",
        rc.depth, rc.limit, rc.target, rc.max_depth, rc.cols,
        rc.output_file, rc.save_all_best,
    )

    searcher = FastSearcher(
        primes=rc.primes,
        depth=rc.depth,
        limit=rc.limit,
        target=rc.target,
        max_depth=rc.max_depth,
        cols=rc.cols,
        output_file=rc.output_file,
        save_all_best=rc.save_all_best,
    )

    searcher.run()

    logger.info("最大値: %d", searcher.max_count)
    logger.info("該当件数: %d", searcher.results)
    logger.info("HLSearch 終了")


if __name__ == "__main__":
    main()