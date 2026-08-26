from pathlib import Path
import pytest
import Config as cfg
import HLSearch
from HLSearch import FastSearcher, build_bit_tables


class TestIntegration:
    """統合テスト: 探索実行から結果の完全性検証まで"""

    def test_solution_path_validation(self, tmp_path: Path, monkeypatch):
        """
        FastSearcher が算出した最適パス(best_paths)を実際に再計算し、
        ビットカウントが max_count と厳密に一致することを事後検証する。
        """
        dummy_file = tmp_path / "integration_shift_path.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(dummy_file))

        primes = cfg.PRIMES[:5]  # [2, 3, 5, 7, 11]
        depth = 4
        cols = 100
        tables = build_bit_tables(primes, cols)

        searcher = FastSearcher(
            primes=primes,
            depth=depth,
            limit=0,
            target=cols,
            max_depth=depth + 2,
            cols=cols,
        )
        searcher.run()

        assert searcher.max_count > 0
        assert len(searcher.best_paths) > 0

        initial_mask = (1 << cols) - 1
        for path in searcher.best_paths:
            mask = initial_mask
            for level, shift in enumerate(path):
                mask &= tables[level][shift]
            # 実際に合成したマスクのビット数が max_count と一致するか
            assert mask.bit_count() == searcher.max_count

    def test_quick_run_with_config_defaults(self, tmp_path: Path, monkeypatch):
        """
        Config のデフォルトパラメータをベースにしつつ、
        depth を小さく調整して短時間で完結する探索実行テスト
        """
        dummy_file = tmp_path / "config_run_shift_path.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(dummy_file))

        quick_depth = 4
        quick_cols = 500
        searcher = FastSearcher(
            primes=cfg.PRIMES,
            depth=quick_depth,
            limit=0,
            target=cfg.TARGET,
            max_depth=cfg.MAX_DEPTH,
            cols=quick_cols,
        )
        searcher.run()

        assert searcher.max_count > 0
        assert searcher.results >= 1
        assert dummy_file.exists()

    def test_main_execution(self, tmp_path: Path, monkeypatch):
        """HLSearch.py をスクリプトとして実行した際の動作をシミュレート検証"""
        import runpy
        dummy_file = tmp_path / "main_shift_path.txt"
        monkeypatch.setattr(cfg, "SHIFT_PATH_FILE", dummy_file)
        # 探索時間を短くするためにパラメータをパッチ
        monkeypatch.setattr(cfg, "DEPTH", 3)
        monkeypatch.setattr(cfg, "COLS", 50)
        monkeypatch.setattr(cfg, "LIMIT", 0)

        # __main__ ブロックを実行
        runpy.run_module("HLSearch", run_name="__main__")
        assert dummy_file.exists()


