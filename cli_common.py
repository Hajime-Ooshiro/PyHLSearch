from __future__ import annotations

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, List, Optional

def build_parser(description: str, include_gpu_flag: bool = False) -> argparse.ArgumentParser:
    """共通のCLI引数パーサを構築する。"""
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

    return parser


def _pick(cli_value: Any, cfg_value: Any, default: Any = None) -> Any:
    """CLI引数が指定されていればそれを、無ければConfig値を、それも無ければdefaultを返す。"""
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default


def resolve_primes(args: argparse.Namespace, cfg: Any) -> List[int]:
    if getattr(args, "primes", None):
        try:
            return [int(x) for x in args.primes.split(",") if x.strip() != ""]
        except ValueError as e:
            raise ValueError(f"--primes の形式が不正です: {args.primes!r}") from e
    return cfg.PRIMES


class ResolvedConfig:
    """CLI引数とConfig.pyの値をマージした実行時設定。"""

    def __init__(self, args: argparse.Namespace, cfg: Any):
        self.primes = resolve_primes(args, cfg)
        self.depth = _pick(args.depth, getattr(cfg, "DEPTH", None))
        self.limit = _pick(args.limit, getattr(cfg, "LIMIT", None))
        self.target = _pick(args.target, getattr(cfg, "TARGET", None))
        self.max_depth = _pick(args.max_depth, getattr(cfg, "MAX_DEPTH", None))
        self.cols = _pick(args.cols, getattr(cfg, "COLS", None))
        self.output_file = Path(_pick(args.output_file, getattr(cfg, "SHIFT_PATH_FILE", None)))
        self.save_all_best = _pick(args.save_all_best, False, False)
        self.show_progress = _pick(args.show_progress, getattr(cfg, "SHOW_PROGRESS", True), True)

        self.log_level = _pick(args.log_level, getattr(cfg, "CONSOLE_LOG_LEVEL", "INFO"))
        self.log_file = _pick(args.log_file, getattr(cfg, "LOG_FILE", "HLSearch.log"))

    def apply_show_progress_override(self, cfg: Any) -> None:
        """cfg.SHOW_PROGRESS を参照しているコード(FastSearcher等)のために、
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
