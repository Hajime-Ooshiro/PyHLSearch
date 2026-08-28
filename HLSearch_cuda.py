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

import Config as cfg
from beam_gpu_core import BeamSearcherGPU, BeamWidthSpec
from HLSearch import make_beam_schedule  # 既存のヘルパーをそのまま利用
from cli_common import build_parser, ResolvedConfig, setup_logging

# モジュールレベルではロガーのハンドルだけ用意し、実際の設定(ハンドラ登録)は
# __main__ 内で CLI引数 / Config.py の値が確定してから行う。
logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_parser(
        description="HLSearch(CUDA版): ビーム法によるシフト探索 (GPU/CuPy, 未導入ならCPUに自動フォールバック)",
        include_gpu_flag=True,
    )
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

    logger.info("HLSearch(CUDA版) 開始 (log file: %s)", rc.log_file)
    logger.debug(
        "実行設定: depth=%s limit=%s target=%s max_depth=%s cols=%s "
        "beam_width=%s output_file=%s save_all_best=%s use_gpu=%s",
        rc.depth, rc.limit, rc.target, rc.max_depth, rc.cols,
        rc.beam_width, rc.output_file, rc.save_all_best, rc.use_gpu,
    )

    searcher = BeamSearcherGPU(
        primes=rc.primes,
        depth=rc.depth,
        limit=rc.limit,
        target=rc.target,
        max_depth=rc.max_depth,
        cols=rc.cols,
        output_file=rc.output_file,
        beam_width=rc.beam_width,
        save_all_best=rc.save_all_best,
        use_gpu=rc.use_gpu,  # cupy/GPUが無ければ自動でCPUにフォールバック。--cpu で強制CPUも可
    )
    searcher.run()

    logger.info("backend: %s", searcher.backend.name)
    logger.info("最大値: %d", searcher.max_count)
    logger.info("該当件数: %d", searcher.results)
    logger.info("HLSearch(CUDA版) 終了")


if __name__ == "__main__":
    main()
