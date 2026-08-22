# HLSearch.py
import logging
import logging.handlers
import numpy as np

import Config as cfg

# --- logging設定(設定値はすべて config.py に集約) ---
LOG_FILE = cfg.LOG_FILE
LOG_MAX_BYTES = cfg.LOG_MAX_BYTES
LOG_BACKUP_COUNT = cfg.LOG_BACKUP_COUNT
CONSOLE_LOG_LEVEL = cfg.CONSOLE_LOG_LEVEL
FILE_LOG_LEVEL = cfg.FILE_LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(cfg.LOG_FORMAT)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(CONSOLE_LOG_LEVEL)
_console_handler.setFormatter(_formatter)

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_file_handler.setLevel(FILE_LOG_LEVEL)
_file_handler.setFormatter(_formatter)

if not logging.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)

COLS = cfg.COLS
IDX = np.arange(1, COLS + 1)  # range(1, 3160)


def shift_array(arr, k):
    n = len(arr)
    if k >= 0:
        if k >= n:
            return np.zeros(n, dtype=arr.dtype)
        result = np.empty(n, dtype=arr.dtype)
        result[:k] = 0
        result[k:] = arr[: n - k]
        return result

    # k < 0: 左シフト
    shift = -k
    if shift >= n:
        return np.zeros(n, dtype=arr.dtype)
    result = np.empty(n, dtype=arr.dtype)
    result[n - shift:] = 0
    result[: n - shift] = arr[shift:]
    return result

def count_zero_mask(mask):
    """真偽値マスク(各列が『これまでの全階層で0』かどうか)からcountを求める"""
    return int(np.count_nonzero(mask))

# 2,3,5,7,11,13,...,1579 の素数リスト(元コードのA2,A3,...,A1579に対応)
# 実体は config.py 側で管理する。
PRIMES = cfg.PRIMES


class SearchState:
    """
    1回の探索(run_search 呼び出し)にひもづく可変状態をまとめたクラス。

    以前は max_count がモジュールグローバル変数だったため、同一プロセス内で
    run_search() を複数回呼ぶと前回の値が残ってしまう(リセット漏れ)リスクや、
    再入/将来的な並列化がしづらいという問題があった。
    探索ごとにこのクラスのインスタンスを1つ作ることで、状態を呼び出し単位に
    閉じ込め、再利用時のバグを構造的に防ぐ。

    Attributes
    ----------
    limit : int
        この件数未満の zero_mask カウントが出た枝は打ち切る(config.LIMIT 相当)。
    target : int
        depth == max_depth のとき、count がこれを超えたら打ち切る特別な閾値
        (config.TARGET 相当)。
    max_depth : int
        depth がこの値と一致するときだけ target による打ち切りを有効にする
        (config.MAX_DEPTH 相当)。
    max_count : int
        探索中に見つかった zero_mask カウントの最大値。実行中に更新される。
    results : list[list[int]]
        max_count を達成した shift_path のリスト。max_count が更新されるとクリアされる。
    """

    def __init__(self, limit, target, max_depth):
        self.limit = limit
        self.target = target
        self.max_depth = max_depth
        self.max_count = 0
        self.results = []


def build_base_array(primes):
    """
    指定した素数リストからA配列を作る。

    後段の処理(search)では「0かどうか」しか使わないため、値そのもの(素数p)
    ではなく bool(idx % p == 1) を格納する。int64(8byte/要素)ではなく
    bool(1byte/要素)にすることでメモリ転送量が1/8になり、shift_array や
    AND演算のコストが下がる。
    """
    return np.array([(IDX % p == 1) for p in primes])

def search(shift_path, primes, depth, base_array, zero_mask, state):
    """
    再帰的に各階層のシフト値を探索する汎用関数。

    count_zero を毎回フルスキャンする代わりに、「これまでの階層すべてで0だった列」
    を表す真偽値マスク zero_mask を引き回し、1階層進めるたびに今回の行の0判定を
    AND するだけにすることで、count計算を O(depth) の毎回全スキャンから
    O(1) の増分更新に落としている(depth回の再帰全体で見ると O(depth^2) -> O(depth))。

    Parameters
    ----------
    shift_path : list[int]
        現在の階層までの各素数に対するシフト値のリスト(最後の要素が今回設定した値)
    primes : list[int]
        探索対象の素数リスト(先頭から順に1階層ずつ対応)
    depth : int
        探索する階層数(= 使用する素数の個数)。可変。
    base_array : np.ndarray
        shape=(depth, COLS) の基準配列(build_base_array(primes[:depth]) で作成)
    zero_mask : np.ndarray
        shape=(COLS,) の真偽値配列。「1つ上の階層までで全て0だった列」を表す。
        呼び出し側は自分のマスクを渡した後、再利用しないこと(内部でANDした
        新しいマスクを作って子呼び出しに渡すため、呼び出し元のものは変更されない)。
    state : SearchState
        limit / target / max_depth といった探索パラメータと、探索中に更新される
        max_count / results をまとめた状態オブジェクト。この1回の探索
        (run_search 呼び出し)の間だけ生存し、再帰全体で共有・更新される。
    """
    level = len(shift_path) - 1  # 今回シフトを設定した階層(0-indexed)
    row_nonzero = shift_array(base_array[level], shift_path[level])
    zero_mask = zero_mask & ~row_nonzero

    count = count_zero_mask(zero_mask)
    logger.debug("depth=%d shift_path=%s count=%s", level + 1, shift_path, count)

    if count < state.limit or count < state.max_count:
        logger.debug("break shift_path=%s count=%d", shift_path, count)
        return  # この枝は打ち切り(子孫を探索しない)

    if level + 1 >= depth:
        """ 最深部ではcountがtargetを超えるのを無効とする """
        if depth == state.max_depth:
            if count > state.target:
                logger.debug("break shift_path=%s count=%d", shift_path, count)
                return

        if count > state.max_count:
            state.max_count = count
            logger.info("max_count=%d", state.max_count)
            state.results.clear()
            state.results.append(shift_path)
            logger.debug("done shift_path=%s count=%d", shift_path, count)
        elif count == state.max_count:
            state.results.append(shift_path)
            logger.debug("done shift_path=%s count=%d", shift_path, count)
        return  # 最深階層に到達

    next_p = primes[level + 1]
    for i in range(next_p):
        search(shift_path + [i], primes, depth, base_array, zero_mask, state)


def run_search(primes, depth, limit=None, target=None, max_depth=None):
    """
    primes[:depth] を使って深さ depth までの探索を実行するエントリポイント。

    limit / target / max_depth を省略した場合は config.py の値(cfg.LIMIT /
    cfg.TARGET / cfg.MAX_DEPTH)を使う。呼び出しごとに新しい SearchState を
    作成するため、同一プロセス内で run_search() を複数回呼んでも前回の
    max_count / results が次回の探索に混入することはない。

    Returns
    -------
    SearchState
        探索完了後の状態(state.results に該当 shift_path のリスト、
        state.max_count に達成した最大カウントが入っている)。
    """
    if depth > len(primes):
        raise ValueError(f"depth={depth} が primes の長さ({len(primes)})を超えています")

    state = SearchState(
        limit=cfg.LIMIT if limit is None else limit,
        target=cfg.TARGET if target is None else target,
        max_depth=cfg.MAX_DEPTH if max_depth is None else max_depth,
    )

    base_array = build_base_array(primes[:depth])
    initial_mask = np.ones(COLS, dtype=bool)

    first_p = primes[0]
    for i in range(first_p):
        search([i], primes, depth, base_array, initial_mask, state)

    return state


if __name__ == "__main__":
    logger.info("HLSearch 開始 (log file: %s)", LOG_FILE)

    DEPTH = cfg.DEPTH  # 使用する素数の個数(可変)。元コードの calc2->calc3->calc5->calc7 相当
    state = run_search(PRIMES, DEPTH)

    logger.info("最大値: %d", state.max_count)
    logger.info("該当件数: %d", len(state.results))
    for shift_path in state.results:
        logger.info("shift_path=%s", shift_path)

    logger.info("HLSearch 終了")