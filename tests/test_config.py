import logging
from pathlib import Path
import pytest
import Config as cfg


def is_prime(n: int) -> bool:
    """素数判定ヘルパー関数"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


class TestConfigConstants:
    """Config.py に定義されている定数の型や値の妥当性を検証するテスト群"""

    def test_debug_type(self):
        assert isinstance(cfg.DEBUG, bool)

    def test_logging_configurations(self):
        assert isinstance(cfg.LOG_DIR, Path)
        assert isinstance(cfg.LOG_FILE, Path)
        assert isinstance(cfg.SHIFT_PATH_FILE, Path)
        assert cfg.LOG_MAX_BYTES > 0
        assert cfg.LOG_BACKUP_COUNT > 0
        assert cfg.CONSOLE_LOG_LEVEL in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        assert cfg.FILE_LOG_LEVEL in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        assert isinstance(cfg.LOG_FORMAT, str)
        assert "%(message)s" in cfg.LOG_FORMAT

    def test_search_parameters(self):
        assert isinstance(cfg.COLS, int) and cfg.COLS > 0
        assert isinstance(cfg.LIMIT, int) and cfg.LIMIT >= 0
        assert isinstance(cfg.TARGET, int) and cfg.TARGET >= 0
        assert isinstance(cfg.DEPTH, int) and cfg.DEPTH > 0
        assert isinstance(cfg.MAX_DEPTH, int) and cfg.MAX_DEPTH >= cfg.DEPTH


class TestConfigPrimes:
    """Config.py の PRIMES リストの正確性を検証するテスト群"""

    def test_primes_not_empty(self):
        assert len(cfg.PRIMES) > 0

    def test_primes_all_valid_primes(self):
        for p in cfg.PRIMES:
            assert is_prime(p), f"{p} は素数ではありません"

    def test_primes_strictly_increasing(self):
        for i in range(len(cfg.PRIMES) - 1):
            assert cfg.PRIMES[i] < cfg.PRIMES[i + 1], (
                f"PRIMES[{i}]={cfg.PRIMES[i]} >= PRIMES[{i+1}]={cfg.PRIMES[i+1]} です"
            )

    def test_primes_first_elements(self):
        expected_first = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert cfg.PRIMES[: len(expected_first)] == expected_first

    def test_depth_within_primes_length(self):
        assert cfg.DEPTH <= len(cfg.PRIMES)
        assert cfg.MAX_DEPTH <= len(cfg.PRIMES)
