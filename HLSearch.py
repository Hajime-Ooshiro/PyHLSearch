# HLSearch.py (numpy高速化版)
import datetime
import logging
import numpy as np
from pathlib import Path

# --- logging設定 ---
LOG_DIR = Path(__file__).resolve().parent
LOG_FILE = LOG_DIR / f"HLFull_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(funcName)s: %(message)s"
)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_formatter)

logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

COLS = 3159
IDX = np.arange(1, COLS + 1)  # 元コードの range(1, 3160) に対応

LIMIT = 400
TARGET = 447


def shift_array(arr, k):
    """配列を右にk個シフトする(numpy版)"""
    n = len(arr)
    if k >= n:
        return np.zeros(n, dtype=arr.dtype)
    result = np.empty(n, dtype=arr.dtype)
    result[:k] = 0
    result[k:] = arr[: n - k]
    return result

def count_zero(arr):
    return int(np.sum(np.all(arr == 0, axis=0)))

# 2,3,5,7,11,13,...,1579 の素数リスト(元コードのA2,A3,...,A1579に対応)
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 
    23, 29, 31, 37, 41, 43, 47, 53, 
    59, 61, 67, 71, 73, 79, 83, 89, 
    97, 101, 103, 107, 109, 113, 127, 131, 
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
    229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479,
    487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
    631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769,
    773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929,
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
    1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193,
    1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307,
    1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459,
    1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579,
]

A2 = np.array([np.where(IDX % p == 1, p, 0) for p in PRIMES[:1]])
A3 = np.array([np.where(IDX % p == 1, p, 0) for p in PRIMES[:2]])
A5 = np.array([np.where(IDX % p == 1, p, 0) for p in PRIMES[:3]])
A7 = np.array([np.where(IDX % p == 1, p, 0) for p in PRIMES[:4]])

max_count = 0

def calc2(key):
    B = A2.copy()
    B[0] = shift_array(A2[0], key[0])

    count = count_zero(B)
    if count < LIMIT:
        logger.debug("break key=%s count=%d", key, count)
        return
    
    for i in range(PRIMES[1]):
        arr = key + [i]
        calc3(arr)

def calc3(key):
    B = A3.copy()
    B[0] = shift_array(A3[0], key[0])
    B[1] = shift_array(A3[1], key[1])

    count = count_zero(B)
    if count < LIMIT:
        logger.debug("break key=%s count=", count)
        return
    
    for i in range(PRIMES[2]):
        arr = key + [i]
        calc5(arr)
    return

def calc5(key):
    B = A5.copy()
    B[0] = shift_array(A5[0], key[0])
    B[1] = shift_array(A5[1], key[1])
    B[2] = shift_array(A7[2], key[2])

    count = count_zero(B)
    if count < LIMIT:
        logger.debug("break key=%s count=%d", list(key), count)
        return
    
    for i in range(PRIMES[3]):
        arr = key + [i]
        calc7(arr)
    return

def calc7(key):
    global max_count
    global shifts
    B = A7.copy()
    B[0] = shift_array(A7[0], key[0])
    B[1] = shift_array(A7[1], key[1])
    B[2] = shift_array(A7[2], key[2])
    B[3] = shift_array(A7[3], key[3])

    count = count_zero(B)
    if count < LIMIT:
        logger.debug("break key=%s count=%d", list(key), count)
        return
    
    if count > max_count:
        max_count = count
        shifts = [key]
        logger.info("max_count=%d", max_count)
        logger.debug("done key=%s count=%d", list(key), count)
        return
    elif count == max_count:
        shifts.append(key)
        logger.debug("done key=%s count=%d", list(key), count)
        return

    if count < max_count:
        logger.debug("break key=%s count=%d", list(key), count)
        return

    if count > TARGET:
        logger.debug("break key=%s count=%d", list(key), count)
        return

shifts = []
if __name__ == "__main__":
    logger.info("HLSearch 開始")
    for i in range(PRIMES[0]):
        calc2([i])

    logger.info("max_count:%d", max_count)
    logger.info("該当件数:%d", len(shifts))
    for shift in shifts:
        logger.info(shift)

    logger.info("HLSearch 終了")
