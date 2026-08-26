# HLSearch_cuda.py — CUDA(CuPy)対応版 BeamSearcher
#
# 使い方はオリジナルの HLSearch.py と同じで、Config.py の設定(PRIMES, DEPTH, LIMIT,
# TARGET, MAX_DEPTH, COLS, BEAM_WIDTH, SHIFT_PATH_FILE 等)をそのまま利用します。
#
# 必要環境:
#   pip install cupy-cuda12x   (お使いのCUDAバージョンに合わせて cupy-cudaXXx を選択)
#   NVIDIA GPU + CUDAドライバ
#
# GPUが無い/cupyが未インストールの場合は自動的にCPU(numpy)実装にフォールバックします
# (正しさの検証や小規模テスト用途向け。本番の速度は出ません)。
#
# 注意: FastSearcher(厳密解法・DFS+分枝限定法)はGPU化していません。
#       深さ優先探索は枝刈り情報を逐次共有する必要があり、素朴な並列化では
#       探索量が増えて逆効果になりやすいためです。厳密解が必要な場合は
#       元の HLSearch.py の FastSearcher をそのまま使ってください。

import logging
import logging.handlers
import time

import Config as cfg
from beam_gpu_core import BeamSearcherGPU, BeamWidthSpec
from HLSearch import make_beam_schedule  # 既存のヘルパーをそのまま利用

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(cfg.LOG_FORMAT)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(cfg.CONSOLE_LOG_LEVEL)
_console_handler.setFormatter(_formatter)
_file_handler = logging.handlers.RotatingFileHandler(
    cfg.LOG_FILE, maxBytes=cfg.LOG_MAX_BYTES, backupCount=cfg.LOG_BACKUP_COUNT, encoding="utf-8",
)
_file_handler.setLevel(cfg.FILE_LOG_LEVEL)
_file_handler.setFormatter(_formatter)
if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)


if __name__ == "__main__":
    logger.info("HLSearch(CUDA版) 開始 (log file: %s)", cfg.LOG_FILE)

    beam_width = getattr(cfg, "BEAM_WIDTH", None)
    if beam_width is None:
        schedule_cfg = getattr(cfg, "BEAM_WIDTH_SCHEDULE", None)
        if schedule_cfg is not None:
            beam_width = make_beam_schedule(
                depth=cfg.DEPTH,
                start_width=schedule_cfg["start"],
                end_width=schedule_cfg["end"],
                mode=schedule_cfg.get("mode", "linear"),
            )
        else:
            beam_width = 100

    searcher = BeamSearcherGPU(
        primes=cfg.PRIMES,
        depth=cfg.DEPTH,
        limit=cfg.LIMIT,
        target=cfg.TARGET,
        max_depth=cfg.MAX_DEPTH,
        cols=cfg.COLS,
        output_file=cfg.SHIFT_PATH_FILE,
        beam_width=beam_width,
        save_all_best=False,
        use_gpu=True,  # cupy/GPUが無ければ自動でCPUにフォールバック
    )
    searcher.run()

    logger.info("backend: %s", searcher.backend.name)
    logger.info("最大値: %d", searcher.max_count)
    logger.info("該当件数: %d", searcher.results)
    logger.info("HLSearch(CUDA版) 終了")
