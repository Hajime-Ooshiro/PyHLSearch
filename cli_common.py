# cli_common.py — HLSearch.py / HLSearch_cuda.py 共通のCLI引数処理
#
# 方針:
#   - Config.py の値をデフォルトとしつつ、コマンドラインから渡された値があれば
#     そちらを優先して上書きする（Config.py 自体は書き換えない）。
#   - 引数を渡さなければ、これまで通り Config.py の設定だけで動く
#     (後方互換性を維持)。
from __future__ import annotations

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

BeamWidthSpec = Union[int, List[int], Callable[[int], int]]


def build_parser(description: str, include_gpu_flag: bool = False) -> argparse.ArgumentParser:
    """共通のCLI引数パーサを構築する。

    include_gpu_flag=True の場合、--gpu / --cpu の排他フラグを追加する
    (HLSearch_cuda.py 用)。
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- 探索パラメータ (未指定ならConfig.pyの値を使用) ---
    parser.add_argument("--depth", type=int, default=None,
                         help="探索する深さ (未指定時: Config.DEPTH)")
    parser.add_argument("--limit", type=int, default=None,
                         help="打ち切り閾値 (未指定時: Config.LIMIT)")
    parser.add_argument("--target", type=int, default=None,
                         help="目標値 (未指定時: Config.TARGET)")
    parser.add_argument("--max-depth", type=int, default=None,
                         help="最大深さ (未指定時: Config.MAX_DEPTH)")
    parser.add_argument("--cols", type=int, default=None,
                         help="列数 (未指定時: Config.COLS)")
    parser.add_argument("--primes", type=str, default=None,
                         help="カンマ区切りの素数リストでConfig.PRIMESを上書き"
                              " (例: 2,3,5,7,11)")

    # --- 出力 ---
    parser.add_argument("-o", "--output-file", type=str, default=None,
                         help="結果出力ファイルパス (未指定時: Config.SHIFT_PATH_FILE)")
    parser.add_argument("--save-all-best", dest="save_all_best", action="store_true",
                         default=None, help="同点の最良解を全て保存する")

    # --- ビーム法 / 厳密解法 ---
    beam_group = parser.add_argument_group("探索アルゴリズム")
    method = beam_group.add_mutually_exclusive_group()
    method.add_argument("--beam", dest="use_beam", action="store_true", default=None,
                         help="ビーム法(近似解)を使う")
    method.add_argument("--exact", dest="use_beam", action="store_false", default=None,
                         help="厳密解法(DFS+分枝限定法)を使う。--exact指定時、"
                              "ビーム幅関連オプションは無視される")
    beam_group.add_argument("--beam-width", type=str, default=None,
                             help="ビーム幅。単一整数(例:100)、または深さごとに"
                                  "カンマ区切り(例:2000,1000,500,200,100,50)")
    beam_group.add_argument("--beam-schedule", type=str, default=None,
                             metavar="START,END[,MODE]",
                             help="ビーム幅を自動生成する。MODEはlinearまたはgeometric"
                                  " (省略時linear)。例: 2000,50,geometric")

    # --- 進捗表示 / ログ ---
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="show_progress", action="store_true",
                           default=None, help="進捗バーを表示する")
    progress.add_argument("--no-progress", dest="show_progress", action="store_false",
                           default=None, help="進捗バーを表示しない")

    parser.add_argument("--log-level", type=str, default=None,
                         choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                         help="コンソールログレベル (未指定時: Config.CONSOLE_LOG_LEVEL)")
    parser.add_argument("--log-file", type=str, default=None,
                         help="ログファイルパス (未指定時: Config.LOG_FILE)")

    if include_gpu_flag:
        gpu_group = parser.add_mutually_exclusive_group()
        gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None,
                                help="GPU(CUDA/cupy)を使う (デフォルト。無ければ自動でCPUにフォールバック)")
        gpu_group.add_argument("--cpu", dest="use_gpu", action="store_false", default=None,
                                help="CPU(numpy)を強制的に使用する")

    return parser


def _pick(cli_value: Any, cfg_value: Any, default: Any = None) -> Any:
    """CLI引数が指定されていればそれを、無ければConfig値を、それも無ければdefaultを返す。"""
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default


def resolve_beam_width(
    args: argparse.Namespace,
    cfg: Any,
    make_beam_schedule: Callable[..., List[int]],
    depth: int,
) -> BeamWidthSpec:
    """優先順位:
       1. --beam-width
       2. --beam-schedule
       3. Config.BEAM_WIDTH
       4. Config.BEAM_WIDTH_SCHEDULE
       5. デフォルト 100
    """
    if getattr(args, "beam_width", None):
        raw = args.beam_width.strip()
        if "," in raw:
            return [int(x) for x in raw.split(",") if x.strip() != ""]
        return int(raw)

    if getattr(args, "beam_schedule", None):
        parts = [p.strip() for p in args.beam_schedule.split(",")]
        if len(parts) < 2:
            raise ValueError(
                "--beam-schedule は START,END[,MODE] の形式で指定してください"
                f" (受け取った値: {args.beam_schedule!r})"
            )
        start_width, end_width = int(parts[0]), int(parts[1])
        mode = parts[2] if len(parts) >= 3 else "linear"
        return make_beam_schedule(depth=depth, start_width=start_width, end_width=end_width, mode=mode)

    beam_width = getattr(cfg, "BEAM_WIDTH", None)
    if beam_width is not None:
        return beam_width

    schedule_cfg = getattr(cfg, "BEAM_WIDTH_SCHEDULE", None)
    if schedule_cfg is not None:
        return make_beam_schedule(
            depth=depth,
            start_width=schedule_cfg["start"],
            end_width=schedule_cfg["end"],
            mode=schedule_cfg.get("mode", "linear"),
        )

    return 100


def resolve_primes(args: argparse.Namespace, cfg: Any) -> List[int]:
    if getattr(args, "primes", None):
        try:
            return [int(x) for x in args.primes.split(",") if x.strip() != ""]
        except ValueError as e:
            raise ValueError(f"--primes の形式が不正です: {args.primes!r}") from e
    return cfg.PRIMES


class ResolvedConfig:
    """CLI引数とConfig.pyの値をマージした実行時設定。"""

    def __init__(self, args: argparse.Namespace, cfg: Any, make_beam_schedule: Callable[..., List[int]]):
        self.primes = resolve_primes(args, cfg)
        self.depth = _pick(args.depth, getattr(cfg, "DEPTH", None))
        self.limit = _pick(args.limit, getattr(cfg, "LIMIT", None))
        self.target = _pick(args.target, getattr(cfg, "TARGET", None))
        self.max_depth = _pick(args.max_depth, getattr(cfg, "MAX_DEPTH", None))
        self.cols = _pick(args.cols, getattr(cfg, "COLS", None))
        self.output_file = Path(_pick(args.output_file, getattr(cfg, "SHIFT_PATH_FILE", None)))
        self.save_all_best = _pick(args.save_all_best, False, False)
        self.use_beam = _pick(getattr(args, "use_beam", None), getattr(cfg, "USE_BEAM_SEARCH", False), False)
        self.use_gpu = _pick(getattr(args, "use_gpu", None), True, True)
        self.show_progress = _pick(args.show_progress, getattr(cfg, "SHOW_PROGRESS", True), True)
        self.beam_width = resolve_beam_width(args, cfg, make_beam_schedule, self.depth)

        self.log_level = _pick(args.log_level, getattr(cfg, "CONSOLE_LOG_LEVEL", "INFO"))
        self.log_file = _pick(args.log_file, getattr(cfg, "LOG_FILE", "HLSearch.log"))

    def apply_show_progress_override(self, cfg: Any) -> None:
        """cfg.SHOW_PROGRESS を参照しているコード(BeamSearcher等)のために、
        CLIで明示的に指定された場合はConfigモジュール側の値も上書きする。
        """
        cfg.SHOW_PROGRESS = self.show_progress


def setup_logging(
    logger_name: str,
    log_file: str,
    console_level: str,
    file_level: Any,
    log_format: str,
    max_bytes: int,
    backup_count: int,
) -> logging.Logger:
    """CLI/Config由来の値でロガーをセットアップする。__main__ から呼び出す想定。"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 再実行(テスト等)でハンドラが重複しないようにクリアしてから追加する
    logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
