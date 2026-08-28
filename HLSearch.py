# HLSearch.py（厳密解法版）
#
# 探索アルゴリズム:
#   --exact : FastSearcher (DFS+分枝限定法。厳密解。CPU(bigint)のみ)
#
import logging
from pathlib import Path
from typing import List, Tuple
import time
from collections import OrderedDict
from tqdm import tqdm
import Config as cfg

from cli_common import build_parser, ResolvedConfig, setup_logging

# モジュールレベルではロガーのハンドルだけ用意し、実際の設定(ハンドラ登録)は
# __main__ 内で CLI引数 / Config.py の値が確定してから行う。
logger = logging.getLogger(__name__)

# bit_table キャッシュ: (primes_tuple, cols) -> bit_tables
_BIT_TABLE_CACHE_SIZE = 4
_bit_table_cache: OrderedDict[tuple[tuple[int, ...], int], List[List[int]]] = OrderedDict()


def _validate_primes(primes: List[int]) -> None:
    for prime in primes:
        if not isinstance(prime, int) or isinstance(prime, bool) or prime < 2:
            raise ValueError("primes には2以上の整数を指定してください")
        divisor = 2
        while divisor * divisor <= prime:
            if prime % divisor == 0:
                raise ValueError(f"素数でない値が含まれています: {prime}")
            divisor += 1
    if any(left >= right for left, right in zip(primes, primes[1:])):
        raise ValueError("primes は重複のない昇順で指定してください")


def build_bit_tables(primes: List[int], cols: int) -> List[List[int]]:
    """
    各素数 p とそのシフト s (0 <= s < p) に対し、
    (idx - s) % p != 1 となる列を残すビットマスクを生成。
    
    キャッシュ機構により、同じ素数リストと cols では計算を再利用する。
    """
    if cols < 0:
        raise ValueError("cols は0以上で指定してください")
    _validate_primes(primes)
    cache_key = (tuple(primes), cols)
    if cache_key in _bit_table_cache:
        _bit_table_cache.move_to_end(cache_key)
        logger.debug("bit_table キャッシュヒット: primes=%d個, cols=%d", len(primes), cols)
        return _bit_table_cache[cache_key]
    
    logger.debug("bit_table 生成開始: primes=%d個, cols=%d", len(primes), cols)
    tables = []
    for p in primes:
        full_blocks, remainder = divmod(cols, p)
        block_mask = (1 << p) - 1
        repeat_mask = (1 << (p * full_blocks)) - 1 if full_blocks else 0
        p_shifts = []
        for s in range(p):
            # 列idx(1始まり)ではなくビット位置(0始まり)の剰余sを除外する。
            pattern = block_mask ^ (1 << s)
            full_mask = pattern * (repeat_mask // block_mask) if full_blocks else 0
            partial_mask = pattern & ((1 << remainder) - 1)
            p_shifts.append(full_mask | (partial_mask << (p * full_blocks)))
        tables.append(p_shifts)
    
    _bit_table_cache[cache_key] = tables
    _bit_table_cache.move_to_end(cache_key)
    while len(_bit_table_cache) > _BIT_TABLE_CACHE_SIZE:
        _bit_table_cache.popitem(last=False)
    logger.debug("bit_table 生成完了 (キャッシュ保存)")
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
        output_file: Path | None = None,
        save_all_best: bool = False,
        show_progress: bool = False,
    ):
        if not primes:
            raise ValueError("primes は空にできません")
        _validate_primes(primes)
        if depth < 1 or depth > len(primes):
            raise ValueError("depth は1以上かつ素数リストの長さ以下で指定してください")
        if max_depth < depth:
            raise ValueError("max_depth は depth 以上で指定してください")
        if cols < 1:
            raise ValueError("cols は1以上で指定してください")
        if limit < 0 or limit > cols:
            raise ValueError("limit は0以上 cols以下で指定してください")
        if target < 0 or target > cols:
            raise ValueError("target は0以上 cols以下で指定してください")

        self.primes = primes[:depth]
        self.depth = depth
        self.limit = limit
        self.target = target
        self.max_depth = max_depth
        self.cols = cols
        self.output_file = output_file or cfg.SHIFT_PATH_FILE
        self.save_all_best = save_all_best
        self.show_progress = show_progress

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
        if self.show_progress:
            self.pbar = tqdm(
                total=self.depth,
                desc="Searching...",
                unit="task",
                dynamic_ncols=True,
            )
        try:
            for s in range(first_p):
                self.shift_path.append(s)
                self._search(0, initial_mask & self.bit_tables[0][s])
                self.shift_path.pop()
        finally:
            if self.show_progress:
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
        if self.show_progress:
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
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
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
    try:
        rc = ResolvedConfig(args, cfg)
    except ValueError as exc:
        parser.error(str(exc))
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
        show_progress=rc.show_progress,
    )

    searcher.run()

    logger.info("最大値: %d", searcher.max_count)
    logger.info("該当件数: %d", searcher.results)
    logger.info("HLSearch 終了")


if __name__ == "__main__":
    main()