import numpy as np
import pytest

from HLSearch import SearchConfig, State, build_shift_table, parse_args, shift_array


def test_parse_args_uses_cpu_by_default_and_can_enable_cuda():
    default_args = parse_args([])
    cuda_args = parse_args(["--cuda"])

    assert not hasattr(default_args, "limit")
    assert not hasattr(default_args, "primes_count")
    assert default_args.cuda is False
    assert cuda_args.cuda is True


def test_state_only_initializes_cuda_when_requested(monkeypatch):
    config = SearchConfig(primes=[2], depth=1, cols=8)
    shift_table = build_shift_table([2], 8)
    cuda_backend = object()
    monkeypatch.setattr(State, "_init_cuda", staticmethod(lambda: cuda_backend))

    cpu_state = State(config, shift_table)
    cuda_state = State(config, shift_table, use_cuda=True)

    assert cpu_state._cuda is None
    assert cuda_state._cuda is cuda_backend


def test_build_shift_table_packs_complemented_shift_rows():
    primes = [3]
    cols = 65
    table = build_shift_table(primes, cols)
    base_row = np.array([(index % 3) == 1 for index in range(1, cols + 1)])

    assert table[0].dtype == np.uint64
    assert table[0].shape == (3, 2)
    for shift in range(primes[0]):
        unpacked = np.unpackbits(table[0][shift].view(np.uint8), bitorder="little")[:cols]
        np.testing.assert_array_equal(unpacked.astype(bool), ~shift_array(base_row, shift))


def test_packed_mask_roundtrip_preserves_multiple_words():
    config = SearchConfig(primes=[2], depth=1, cols=65)
    state = State(config, build_shift_table([2], config.cols))
    mask = np.array([0x0123456789ABCDEF, 0x0000000000000001], dtype=np.uint64)

    restored = state._int_to_mask(state._mask_to_int(mask), config.cols)

    np.testing.assert_array_equal(restored, mask)


def test_search_uses_prime_ranges_as_shift_candidates():
    primes = [2, 3]
    cols = 8
    shift_table = build_shift_table(primes, cols)

    config = SearchConfig(
        primes=primes,
        depth=2,
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


def test_search_tries_shift_candidates_in_descending_order():
    primes = [3]
    cols = 4
    config = SearchConfig(
        primes=primes,
        depth=1,
        target=cols,
        max_depth=1,
        cols=cols,
    )
    shift_table = [np.full((primes[0], 1), np.iinfo(np.uint64).max, dtype=np.uint64)]

    state = State(config, shift_table)
    state.run()

    assert state.shifts == [[2], [1], [0]]


def test_search_keeps_target_and_maximum_paths_without_duplicates():
    config = SearchConfig(primes=[2], depth=1, target=1, max_depth=3, cols=2)
    shift_table = [np.array([[0b11], [0b01]], dtype=np.uint64)]

    state = State(config, shift_table)
    state.run()

    assert state.results == 1
    assert state.target_shifts == [[1]]
    assert state.max_shifts == [[0]]
    assert state.shifts == [[1], [0]]


def test_search_with_zero_depth_finishes_without_exploring():
    config = SearchConfig(
        primes=[],
        depth=0,
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
        State(config, [np.zeros((1, 1), dtype=np.uint64)])


def test_state_rejects_unpacked_shift_table():
    config = SearchConfig(primes=[2], depth=1, cols=8)

    with pytest.raises(ValueError, match=r"shift_table\[0\]"):
        State(config, [np.zeros((2, 1), dtype=bool)])


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


def test_count_nonzero_uses_numpy_by_default():
    config = SearchConfig(primes=[2], depth=1, cols=8)
    state = State(config, build_shift_table([2], 8))
    mask = np.array([0b01001101], dtype=np.uint64)

    assert state._count_nonzero(mask) == 4
