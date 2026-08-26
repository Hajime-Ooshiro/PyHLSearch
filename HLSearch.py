# HLSearch.py (改善版)
import logging
import logging.handlers
from pathlib import Path
from typing import List, Tuple, Union, Callable, Optional
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


# ビーム幅の指定方法:
#   - int                 : 全深さで一定のビーム幅
#   - List[int]           : 深さごとのビーム幅（levelインデックスで参照。
#                            リストが深さより短い場合は最後の値を以降も使い回す）
#   - Callable[[int], int]: level(0-indexed)を受け取りビーム幅を返す関数
BeamWidthSpec = Union[int, List[int], Callable[[int], int]]


def make_beam_schedule(
    depth: int,
    start_width: int,
    end_width: int,
    mode: str = "linear",
) -> List[int]:
    """深さごとのビーム幅リストを生成するヘルパー。

    mode="linear"    : start_width -> end_width まで等差数列で変化
    mode="geometric"  : start_width -> end_width まで等比数列で変化
                        （序盤は広く探索し、終盤は急速に絞り込む、あるいはその逆、
                        といったチューニングがしやすい）

    例:
        # 序盤(浅い階層)は広く、終盤(深い階層)は狭くする
        make_beam_schedule(depth=20, start_width=2000, end_width=50, mode="geometric")

        # 序盤は狭く、終盤に向けて広げる
        make_beam_schedule(depth=20, start_width=50, end_width=2000, mode="linear")
    """
    if depth <= 1:
        return [start_width]

    widths: List[int] = []
    if mode == "linear":
        step = (end_width - start_width) / (depth - 1)
        for i in range(depth):
            widths.append(max(1, round(start_width + step * i)))
    elif mode == "geometric":
        s = max(1, start_width)
        e = max(1, end_width)
        ratio = (e / s) ** (1 / (depth - 1))
        for i in range(depth):
            widths.append(max(1, round(s * (ratio ** i))))
    else:
        raise ValueError(f"Unknown mode: {mode!r} (use 'linear' or 'geometric')")
    return widths


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


class BeamSearcher:
    """ビーム法（近似解法）。

    深さ（レベル）ごとに、全ての生存パス（ビーム）を一斉に子へ展開し、
    残存ビット数の多い順に上位 beam_width 個だけをグローバルに残して
    次の深さへ進む。FastSearcher の分枝限定法と異なり、
    「あるノードの兄弟内での上位」ではなく「その深さ全体での上位」を
    基準に打ち切るため、探索幅を明示的かつ一定に制御できる。

    速度と引き換えに、打ち切られた経路の先に真の最適解があっても
    見逃す可能性がある（厳密解法ではない）。
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
    ):
        """
        beam_width: 各深さで残す上位パス数。
            - int              : 全深さ共通の固定幅
            - List[int]        : 深さごとの幅（levelインデックスで参照。
                                  リストが depth より短い場合は末尾の値を使い回す）
            - Callable[[int], int]: level(0-indexed)を受け取り幅を返す関数
        """
        self.primes = primes[:depth]
        self.depth = depth
        self.limit = limit
        self.target = target
        self.max_depth = max_depth
        self.cols = cols
        self.output_file = output_file
        self.beam_width = beam_width
        self.save_all_best = save_all_best

        self.bit_tables = build_bit_tables(self.primes, cols)

        self.max_count = 0
        self.results = 0
        self.nodes_searched = 0
        self.best_paths: List[List[int]] = []

    def _resolve_beam_width(self, level: int) -> int:
        """指定した深さ(level, 0-indexed)で使うビーム幅を決定する。"""
        bw = self.beam_width
        if callable(bw):
            width = bw(level)
        elif isinstance(bw, (list, tuple)):
            if len(bw) == 0:
                raise ValueError("beam_width のリストが空です")
            width = bw[level] if level < len(bw) else bw[-1]
        else:
            width = bw

        if width < 1:
            raise ValueError(f"beam_width は1以上である必要があります (level={level}, width={width})")
        return width

    def run(self) -> "BeamSearcher":
        start_time = time.time()
        initial_mask = (1 << self.cols) - 1

        # ビーム = (count, mask, path) のリスト
        first_p = self.primes[0]
        beam: List[Tuple[int, int, List[int]]] = []
        for s in range(first_p):
            mask = initial_mask & self.bit_tables[0][s]
            count = mask.bit_count()
            self.nodes_searched += 1
            if count >= self.limit:
                beam.append((count, mask, [s]))

        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[: self._resolve_beam_width(0)]

        if cfg.SHOW_PROGRESS:
            self.pbar = tqdm(
                total=self.depth,
                desc="Beam Searching...",
                unit="level",
                dynamic_ncols=True,
            )
            self.pbar.update(1)

        for level in range(1, self.depth):
            next_p = self.primes[level]
            table_next = self.bit_tables[level]
            is_last_level = (level + 1 >= self.depth)

            next_beam: List[Tuple[int, int, List[int]]] = []
            for count, mask, path in beam:
                for s in range(next_p):
                    next_mask = mask & table_next[s]
                    next_count = next_mask.bit_count()
                    self.nodes_searched += 1

                    if next_count < self.limit:
                        continue
                    if is_last_level and self.depth == self.max_depth and next_count > self.target:
                        continue

                    next_beam.append((next_count, next_mask, path + [s]))

            if not next_beam:
                logger.warning("ビームが空になりました (level=%d)。探索を打ち切ります。", level)
                break

            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[: self._resolve_beam_width(level)]

            if cfg.SHOW_PROGRESS:
                self.pbar.update(1)

        if cfg.SHOW_PROGRESS:
            self.pbar.close()

        # 最終ビームから最良解を集計
        if beam:
            self.max_count = max(c for c, _, _ in beam)
            for count, _, path in beam:
                if count == self.max_count:
                    self.results += 1
                    if self.save_all_best or not self.best_paths:
                        self.best_paths.append(path)
                    if not self.save_all_best:
                        break

        self.save_best_paths()

        elapsed = time.time() - start_time
        widths_used = [self._resolve_beam_width(lv) for lv in range(self.depth)]
        logger.info(
            "ビーム探索完了: 所要時間=%.2f秒, 探索ノード数=%d (%.2f nodes/s), beam_width=%s",
            elapsed,
            self.nodes_searched,
            self.nodes_searched / max(elapsed, 1e-6),
            widths_used if len(set(widths_used)) > 1 else widths_used[0],
        )
        return self

    def save_best_paths(self) -> None:
        """最良解をファイルに出力する"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"max_count:{self.max_count}\n")
            f.write(f"results_count:{self.results}\n")
            for path in self.best_paths:
                f.write(f"{path}\n")


def main() -> None:
    parser = build_parser(description="HLSearch: ビーム法/厳密解法によるシフト探索 (CPU版)")
    args = parser.parse_args()
    rc = ResolvedConfig(args, cfg, make_beam_schedule)
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
        "実行設定: depth=%s limit=%s target=%s max_depth=%s cols=%s "
        "beam=%s beam_width=%s output_file=%s save_all_best=%s",
        rc.depth, rc.limit, rc.target, rc.max_depth, rc.cols,
        rc.use_beam, rc.beam_width, rc.output_file, rc.save_all_best,
    )

    if rc.use_beam:
        # BEAM_WIDTH は int / List[int] / Callable[[int], int] のいずれでもよい。
        # CLIからは --beam-width / --beam-schedule で指定でき、未指定なら
        # Config.py の BEAM_WIDTH / BEAM_WIDTH_SCHEDULE、それも無ければ100を使う。
        searcher = BeamSearcher(
            primes=rc.primes,
            depth=rc.depth,
            limit=rc.limit,
            target=rc.target,
            max_depth=rc.max_depth,
            cols=rc.cols,
            output_file=rc.output_file,
            beam_width=rc.beam_width,
            save_all_best=rc.save_all_best,
        )
    else:
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