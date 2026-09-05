from HLSearch import SearchConfig, State, build_shift_table


def test_search_uses_prime_ranges_as_shift_candidates():
    primes = [2, 3]
    cols = 8
    shift_table = build_shift_table(primes, cols)

    config = SearchConfig(
        primes=primes,
        depth=2,
        limit=0,
        target=cols + 1,
        max_depth=2,
        cols=cols,
    )

    state = State(config, shift_table)
    state.run()

    assert state.node_count == 8
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
