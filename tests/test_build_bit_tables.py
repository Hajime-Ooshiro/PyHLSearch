import pytest
from typing import List
from HLSearch import build_bit_tables


class TestBuildBitTables:
    """build_bit_tables 関数の単体テスト"""

    def test_return_structure_and_lengths(self):
        primes = [2, 3, 5, 7]
        cols = 30
        tables = build_bit_tables(primes, cols)

        assert len(tables) == len(primes)
        for i, p in enumerate(primes):
            assert len(tables[i]) == p

    def test_specific_masks_p2(self):
        """p=2, cols=6 の場合のビットマスク手計算値検証"""
        primes = [2]
        cols = 6
        tables = build_bit_tables(primes, cols)

        # s=0: (idx-0)%2 != 1 => idx=2, 4, 6 が 1 (bits: 1, 3, 5) -> 0b101010 = 42
        assert tables[0][0] == 42
        # s=1: (idx-1)%2 != 1 => idx=1, 3, 5 が 1 (bits: 0, 2, 4) -> 0b010101 = 21
        assert tables[0][1] == 21

    def test_specific_masks_p3(self):
        """p=3, cols=6 の場合のビットマスク手計算値検証"""
        primes = [3]
        cols = 6
        tables = build_bit_tables(primes, cols)

        # s=0: idx%3 != 1 => idx=2, 3, 5, 6 が 1 (bits: 1, 2, 4, 5) -> 0b110110 = 54
        assert tables[0][0] == 54
        # s=1: (idx-1)%3 != 1 => idx=1, 3, 4, 6 が 1 (bits: 0, 2, 3, 5) -> 0b101101 = 45
        assert tables[0][1] == 45
        # s=2: (idx-2)%3 != 1 => idx=1, 2, 4, 5 が 1 (bits: 0, 1, 3, 4) -> 0b011011 = 27
        assert tables[0][2] == 27

    @pytest.mark.parametrize("primes,cols", [
        ([2, 3], 10),
        ([2, 3, 5], 30),
        ([7, 11], 50),
    ])
    def test_bit_level_exact_definition(self, primes: List[int], cols: int):
        """すべてのビット位置において定義通り (idx - s) % p != 1 と一致するか網羅検証"""
        tables = build_bit_tables(primes, cols)

        for p_idx, p in enumerate(primes):
            for s in range(p):
                mask = tables[p_idx][s]
                for idx in range(1, cols + 1):
                    bit = (mask >> (idx - 1)) & 1
                    expected_bit = 1 if (idx - s) % p != 1 else 0
                    assert bit == expected_bit, (
                        f"p={p}, s={s}, idx={idx} において bit={bit} != expected={expected_bit}"
                    )

    def test_empty_primes(self):
        """素数リストが空の場合の動作"""
        tables = build_bit_tables([], 10)
        assert tables == []

    def test_cols_zero(self):
        """cols=0 の場合、全マスクが 0 になること"""
        primes = [2, 3]
        tables = build_bit_tables(primes, 0)
        assert len(tables) == 2
        assert tables[0] == [0, 0]
        assert tables[1] == [0, 0, 0]

    def test_cols_one(self):
        """cols=1 の場合"""
        primes = [2, 3]
        tables = build_bit_tables(primes, 1)
        # p=2: s=0 -> (1-0)%2 = 1 == 1 -> bit0 is 0 => mask=0
        #      s=1 -> (1-1)%2 = 0 != 1 -> bit0 is 1 => mask=1
        assert tables[0] == [0, 1]
        # p=3: s=0 -> (1-0)%3 = 1 == 1 -> mask=0
        #      s=1 -> (1-1)%3 = 0 != 1 -> mask=1
        #      s=2 -> (1-2)%3 = (-1)%3 = 2 != 1 -> mask=1
        assert tables[1] == [0, 1, 1]
