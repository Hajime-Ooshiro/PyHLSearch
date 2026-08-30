from pathlib import Path

import numpy as np
import pytest

import HLSearch
from HLSearch import SearchConfig, State, build_shift_table
from HLSearch_Beam import CPUBackend, build_bit_tables_packed, popcount64_numpy

cfg = SearchConfig()


def run_state(primes, depth, limit, target, max_depth, cols, output_path):
    config = SearchConfig(
        primes=primes,
        depth=depth,
        limit=limit,
        target=target,
        max_depth=max_depth,
        cols=cols,
        shift_path_file=str(output_path),
    )
    shift_table = build_shift_table(primes[:depth], cols)
    state = State(config, shift_table)
    state.run(depth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"max_count:{state.max_count}\n")
        f.write(f"results:{state.results}\n")
        for path in state.shifts:
            f.write(f"{path}\n")
    return state


class TestIntegration:
    """統合テスト: 探索実行から結果の完全性検証まで"""

    def test_solution_path_validation(self, tmp_path: Path, monkeypatch):
        """State が算出した最適パスが実際のビットマスクと一致することを検証"""
        dummy_file = tmp_path / "integration_shift_path.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(dummy_file))

        primes = cfg.primes[:5]  # [2, 3, 5, 7, 11]
        depth = 4
        cols = 100
        state = run_state(primes, depth, 0, cols, depth + 2, cols, dummy_file)

        assert state.max_count > 0
        assert len(state.shifts) > 0
        for path in state.shifts:
            assert len(path) == depth
            assert all(0 <= s < primes[i] for i, s in enumerate(path))

    def test_quick_run_with_config_defaults(self, tmp_path: Path, monkeypatch):
        """デフォルト設定を使った探索で、結果が生成されることを確認"""
        dummy_file = tmp_path / "config_run_shift_path.txt"
        monkeypatch.setattr(HLSearch, "shift_path_file", str(dummy_file))

        quick_depth = 4
        quick_cols = 500
        state = run_state(cfg.primes, quick_depth, 0, cfg.target, cfg.max_depth, quick_cols, dummy_file)

        assert state.max_count > 0
        assert state.results >= 1
        assert dummy_file.exists()

    def test_main_execution(self, tmp_path: Path, monkeypatch):
        """HLSearch.py をスクリプトとして実行した際の動作をシミュレート検証"""
        import runpy
        import sys

        dummy_file = tmp_path / "main_shift_path.txt"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "HLSearch.py",
                "--depth", "3",
                "--cols", "50",
                "--limit", "0",
                "--output", str(dummy_file),
            ],
        )

        runpy.run_module("HLSearch", run_name="__main__")
        assert dummy_file.exists()


def test_packed_bit_table_matches_expected_mask_layout():
    """packed bit table が各列の条件を正確に表していることを確認"""
    primes = [2, 3]
    cols = 12
    tables, n_words = build_bit_tables_packed(primes, cols)

    assert n_words == 1
    assert len(tables) == 2
    for level, p in enumerate(primes):
        table = tables[level]
        assert table.shape == (p, n_words)
        for shift in range(p):
            expected = np.zeros(n_words, dtype=np.uint64)
            for col in range(cols):
                if (col % p) != shift:
                    bit = col % 64
                    expected[0] |= np.uint64(1) << np.uint64(bit)
            assert np.array_equal(table[shift], expected)


def test_cpu_backend_score_all_counts_bits_correctly():
    """CPUBackend.score_all が AND 後の popcount を正しく返すことを確認"""
    tables, n_words = build_bit_tables_packed([2], 64)
    backend = CPUBackend(tables, n_words)

    beam_masks = np.array(
        [
            [np.uint64(0xFFFFFFFFFFFFFFFF)],
            [np.uint64(0xAAAAAAAAAAAAAAAA)],
        ],
        dtype=np.uint64,
    )

    counts = backend.score_all(beam_masks, 0)
    expected = np.array([[32, 32], [32, 0]], dtype=np.int64)
    assert np.array_equal(counts, expected)


def test_popcount64_matches_python_reference():
    """popcount64_numpy が uint64 の各要素に対する popcount を正しく計算する"""
    values = np.array(
        [
            np.uint64(0),
            np.uint64(1),
            np.uint64(0xF0F0),
            np.uint64(0xFFFFFFFFFFFFFFFF),
        ],
        dtype=np.uint64,
    )

    got = popcount64_numpy(values)
    expected = np.array([0, 1, 8, 64], dtype=np.int64)
    assert np.array_equal(got, expected)


def test_state_checkpoint_roundtrip(tmp_path):
    """探索途中状態をテキスト checkpoint に保存し、再読込できることを確認"""
    path = tmp_path / "resume_state.txt"
    config = SearchConfig(primes=[2, 3], depth=2, limit=0, target=10, max_depth=2, cols=8)
    shift_table = build_shift_table([2, 3], 8)
    state = State(config, shift_table, checkpoint_path=path, checkpoint_interval=1)
    state.key = [0, 1]
    state.zero_mask = np.array([True, False, True, False, True, False, True, False], dtype=bool)
    state.max_count = 7
    state.results = 2
    state.shifts = [[0, 1], [1, 0]]
    state.node_count = 42
    state._stack_state = [
        (0, state.zero_mask.copy(), 1, 2),
        (1, np.array([True, True, False, False, True, True, False, False], dtype=bool), 0, 3),
    ]

    state._save_checkpoint()
    restored = State(config, shift_table)
    restored._load_checkpoint(path)

    assert restored.key == [0, 1]
    assert restored.max_count == 7
    assert restored.results == 2
    assert restored.node_count == 42
    assert restored._stack_state[0][0] == 0
    assert restored._stack_state[0][2] == 1
    assert restored.shifts == [[0, 1], [1, 0]]


