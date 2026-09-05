import numpy as np

from HLSearch import SearchConfig, State, build_shift_table


def test_search_uses_nums_values_as_shift_candidates():
    primes = [2, 3]
    nums = [[1], [2]]
    cols = 8
    shift_table = build_shift_table(primes, cols)

    expected_count = int(
        np.count_nonzero(shift_table[0][nums[0][0]] & shift_table[1][nums[1][0]])
    )
    config = SearchConfig(
        primes=primes,
        nums=nums,
        depth=2,
        limit=0,
        target=expected_count,
        max_depth=2,
        cols=cols,
    )

    state = State(config, shift_table)
    state.run()

    assert state.node_count == 2
    assert state.shifts == [[1, 2]]
