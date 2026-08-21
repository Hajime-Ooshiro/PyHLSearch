# HLSearch.py (numpy高速化版)
import logging
import logging.handlers
import numpy as np
from pathlib import Path

# --- logging設定 ---
LOG_DIR = Path(__file__).resolve().parent
# ファイル名は固定(HLSearch.log)。サイズが上限を超えたら
# HLSearch.log.1, HLSearch.log.2, ... にローテーションする。
LOG_FILE = LOG_DIR / "HLSearch.log"

# 1ファイルあたりの最大サイズ(bytes)。超えたらローテーション。
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
# 保持する世代数(HLSearch.log.1 ~ HLSearch.log.<LOG_BACKUP_COUNT>)
LOG_BACKUP_COUNT = 100

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(funcName)s: %(message)s"
)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_formatter)

logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

COLS = 3159
IDX = np.arange(1, COLS + 1)  # range(1, 3160) 

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

def count_zero_mask(mask):
    """真偽値マスク(各列が『これまでの全階層で0』かどうか)からcountを求める"""
    return int(np.count_nonzero(mask))

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
max_count = 0

def build_A(primes):
    """
    指定した素数リストからA配列を作る。

    後段の処理(search)では「0かどうか」しか使わないため、値そのもの(素数p)
    ではなく bool(idx % p == 1) を格納する。int64(8byte/要素)ではなく
    bool(1byte/要素)にすることでメモリ転送量が1/8になり、shift_array や
    AND演算のコストが下がる。
    """
    return np.array([(IDX % p == 1) for p in primes])

def search(key, primes, depth, A, zero_mask, results=None):
    """
    再帰的に各階層のシフト値を探索する汎用関数。

    count_zero を毎回フルスキャンする代わりに、「これまでの階層すべてで0だった列」
    を表す真偽値マスク zero_mask を引き回し、1階層進めるたびに今回の行の0判定を
    AND するだけにすることで、count計算を O(depth) の毎回全スキャンから
    O(1) の増分更新に落としている(depth回の再帰全体で見ると O(depth^2) -> O(depth))。

    Parameters
    ----------
    key : list[int]
        現在の階層までの各素数に対するシフト値のリスト(最後の要素が今回設定した値)
    primes : list[int]
        探索対象の素数リスト(先頭から順に1階層ずつ対応)
    depth : int
        探索する階層数(= 使用する素数の個数)。可変。
    A : np.ndarray
        shape=(depth, COLS) の基準配列(build_A(primes[:depth]) で作成)
    zero_mask : np.ndarray
        shape=(COLS,) の真偽値配列。「1つ上の階層までで全て0だった列」を表す。
        呼び出し側は自分のマスクを渡した後、再利用しないこと(内部でANDした
        新しいマスクを作って子呼び出しに渡すため、呼び出し元のものは変更されない)。
    results : list, optional
        LIMIT以上のcountが出たkeyとcountを記録するリスト。Noneなら記録しない。
    """
    global max_count
    level = len(key) - 1  # 今回シフトを設定した階層(0-indexed)
    row_nonzero = shift_array(A[level], key[level])  # True = その素数の倍数位置
    zero_mask = zero_mask & ~row_nonzero

    count = count_zero_mask(zero_mask)
    logger.debug("depth=%d key=%s count=%s", level + 1, key, count)

    if count < LIMIT:
        logger.debug(("break", list(key), count))
        return  # この枝は打ち切り(子孫を探索しない)

    if count < max_count:
        logger.debug(("break", list(key), count))
        return  # この枝は打ち切り(子孫を探索しない)

    if level + 1 >= depth:
        if count > max_count:
            # if count > TARGET:
            #     logger.debug(("break", list(key), count))
            #     return
            max_count = count
            logger.info("max_count=%d", max_count)
            results = [key]
            logger.debug(("done", list(key), count))
        if count == max_count:
            results.append(key)
            logger.debug(("done", list(key), count))
        return  # 最深階層に到達

    next_p = primes[level + 1]
    for i in range(next_p):
        search(key + [i], primes, depth, A, zero_mask, results)


def run_search(primes, depth):
    """primes[:depth] を使って深さ depth までの探索を実行するエントリポイント"""
    if depth > len(primes):
        raise ValueError(f"depth={depth} が primes の長さ({len(primes)})を超えています")

    A = build_A(primes[:depth])
    results = []
    initial_mask = np.ones(COLS, dtype=bool)

    first_p = primes[0]
    for i in range(first_p):
        search([i], primes, depth, A, initial_mask, results)

    return results


if __name__ == "__main__":
    logger.info("HLSearch 開始 (log file: %s)", LOG_FILE)

    DEPTH = 8  # 使用する素数の個数(可変)。元コードの calc2->calc3->calc5->calc7 相当
    results = run_search(PRIMES, DEPTH)

    logger.info("最大値: %d", max_count)
    logger.info("該当件数: %d",len(results))
    for key in results:
        logger.info("key=%s", key)

    logger.info("HLSearch 終了")