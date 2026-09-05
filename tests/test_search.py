import numpy as np
import pytest

from HLSearch import SearchConfig, State, build_shift_table


def test_search_uses_prime_ranges_as_shift_candidates():
    primes = [2, 3]
    cols = 8
    shift_table = build_shift_table(primes, cols)

    config = SearchConfig(
        primes=primes,
        depth=2,
        limit=0,
        target=2,
        max_depth=2,
        cols=cols,
    )

    state = State(config, shift_table)
    state.run()

    assert state.node_count == 8
    assert state.shifts
    assert state.results == len(state.shifts)
    assert all(len(path) == config.depth for path in state.shifts)
    assert all(0 <= shift < primes[level] for path in state.shifts for level, shift in enumerate(path))


def test_search_with_zero_depth_finishes_without_exploring():
    config = SearchConfig(
        primes=[],
        depth=0,
        limit=0,
        target=0,
        max_depth=0,
        cols=8,
    )
    state = State(config, [])

    result = state.run()

    assert result is state
    assert state.node_count == 0
    assert state.key == []
    assert state.max_count == 0
    assert state.results == 0
    assert state.shifts == []


def test_state_rejects_inconsistent_shift_table():
    config = SearchConfig(primes=[2], depth=1, cols=8)

    with pytest.raises(ValueError, match=r"shift_table\[0\]"):
        State(config, [np.zeros((1, 8), dtype=bool)])


def test_state_rejects_checkpoint_with_different_settings(tmp_path):
    path = tmp_path / "state.json"
    config = SearchConfig(primes=[2], depth=1, cols=8)
    shift_table = build_shift_table([2], 8)
    state = State(config, shift_table, checkpoint_path=path)
    state._save_checkpoint()

    changed_config = SearchConfig(primes=[3], depth=1, cols=8)
    changed_state = State(changed_config, build_shift_table([3], 8))

    with pytest.raises(ValueError, match="探索設定"):
        changed_state._load_checkpoint(path)


def test_count_nonzero_matches_numpy_when_cuda_is_unavailable():
    config = SearchConfig(primes=[2], depth=1, cols=8)
    state = State(config, build_shift_table([2], 8))
    mask = np.array([True, False, True, True, False, False, True, False])

    state._cuda = None

    assert state._count_nonzero(mask) == 4
