import itertools
from pathlib import Path
import pytest
import HLSearch
from HLSearch import FastSearcher, build_bit_tables


def solve_brute_force(primes: list[int], depth: int, cols: int):
    """
    ブルートフォース（全探索）で最大ビット数とその達成パス数を計算する参照実装
    """
    active_primes = primes[:depth]
    tables = build_bit_tables(active_primes, cols)
    initial_mask = (1 << cols) - 1

    shift_choices = [list(range(p)) for p in active_primes]
    max_count = -1
    best_paths = []

    for path in itertools.product(*shift_choices):
        mask = initial_mask
        for level, s in enumerate(path):
            mask &= tables[level][s]
        cnt = mask.bit_count()
        if cnt > max_count:
            max_count = cnt
            best_paths = [list(path)]
        elif cnt == max_count:
            best_paths.append(list(path))

    return max_count, len(best_paths), best_paths


class TestFastSearcherInitialization:
    """FastSearcher 初期化のテスト"""

    def test_init_properties(self):
        primes = [2, 3, 5, 7, 11]
        depth = 3
        limit = 5
        target = 10
        max_depth = 10
        cols = 20

        searcher = FastSearcher(primes, depth, limit, target, max_depth, cols)

        assert searcher.primes == [2, 3, 5]
        assert searcher.depth == 3
        assert searcher.limit == 5
        assert searcher.target == 10
        assert searcher.max_depth == 10
        assert searcher.cols == 20
        assert len(searcher.bit_tables) == 3
        assert searcher.max_count == 0
        assert searcher.results == 0
        assert searcher.nodes_searched == 0
        assert searcher.shift_path == []
        assert searcher.best_paths == []


class TestFastSearcherExecution:
    """FastSearcher の探索ロジックおよび結果の正確性テスト"""

    @pytest.mark.parametrize("depth,cols", [
        (2, 15),
        (3, 20),
        (3, 35),
    ])
    def test_search_matches_brute_force(self, depth: int, cols: int, tmp_path: Path, monkeypatch):
        """ブルートフォースの正解と FastSearcher の結果が完全一致するか検証"""
        primes = [2, 3, 5, 7, 11]
        dummy_file = tmp_path / "test_shift_path.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(dummy_file))

        expected_max, expected_results_count, expected_paths = solve_brute_force(primes, depth, cols)

        searcher = FastSearcher(
            primes=primes,
            depth=depth,
            limit=0,
            target=cols,
            max_depth=depth + 5,
            cols=cols,
        )
        returned_self = searcher.run()

        assert returned_self is searcher
        assert searcher.max_count == expected_max
        assert searcher.results == expected_results_count
        assert searcher.nodes_searched > 0
        # best_paths に含まれるパスの正当性検証
        assert len(searcher.best_paths) >= 1
        for path in searcher.best_paths:
            assert len(path) == depth
            assert path in expected_paths

    def test_pruning_by_limit(self, tmp_path: Path, monkeypatch):
        """limit による枝刈りの動作検証"""
        dummy_file = tmp_path / "test_shift_path.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(dummy_file))

        primes = [2, 3, 5]
        depth = 3
        cols = 30

        # limit = 0 (枝刈り最小)
        searcher_nolimit = FastSearcher(primes, depth, limit=0, target=cols, max_depth=10, cols=cols)
        searcher_nolimit.run()

        # limit = 最大値ちょうど (枝刈り有効)
        searcher_limited = FastSearcher(primes, depth, limit=searcher_nolimit.max_count, target=cols, max_depth=10, cols=cols)
        searcher_limited.run()

        assert searcher_limited.max_count == searcher_nolimit.max_count
        assert searcher_limited.nodes_searched <= searcher_nolimit.nodes_searched

        # limit が最大値より大きい場合 (すべての枝が刈られる)
        searcher_overlimit = FastSearcher(primes, depth, limit=searcher_nolimit.max_count + 1, target=cols, max_depth=10, cols=cols)
        searcher_overlimit.run()

        assert searcher_overlimit.max_count == 0
        assert searcher_overlimit.results == 0

    def test_pruning_by_target_at_max_depth(self, tmp_path: Path, monkeypatch):
        """depth == max_depth かつ count > target の枝刈り条件の検証"""
        dummy_file = tmp_path / "test_shift_path.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(dummy_file))

        primes = [2, 3]
        depth = 2
        cols = 20

        # まず制約なしで探索
        searcher_normal = FastSearcher(primes, depth, limit=0, target=cols, max_depth=10, cols=cols)
        searcher_normal.run()
        normal_max = searcher_normal.max_count

        # target を normal_max - 1 に設定し、max_depth == depth とする
        # これにより count > target となる最大値の枝は除外され、target 以下の解のみ残るか確認
        searcher_target = FastSearcher(primes, depth, limit=0, target=normal_max - 1, max_depth=depth, cols=cols)
        searcher_target.run()

        assert searcher_target.max_count <= normal_max - 1


class TestFastSearcherFileOperations:
    """shift_path.txt の出力機能に関するテスト"""

    def test_append_shift_path_file(self, tmp_path: Path, monkeypatch):
        out_file = tmp_path / "output_shifts.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(out_file))

        primes = [2, 3]
        searcher = FastSearcher(primes, depth=2, limit=0, target=20, max_depth=5, cols=20)
        searcher.run()

        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(content) >= 1
        assert content[0] == f"max_count:{searcher.max_count}"
        for path_line in content[1:]:
            # パス行が [s0, s1] の形式になっているか
            assert path_line.startswith("[") and path_line.endswith("]")
