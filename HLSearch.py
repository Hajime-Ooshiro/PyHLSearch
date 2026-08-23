# HLSearch.py
import logging
import logging.handlers
from typing import List, Optional

import numpy as np
import numpy.typing as npt

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

if not logger.handlers:
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)

COLS: int = cfg.COLS
IDX: npt.NDArray[np.int64] = np.arange(1, COLS + 1)  # range(1, 3160)

SHIFT_PATH_FILE = cfg.SHIFT_PATH_FILE


def clear_shift_path_file() -> None:
    """shift_path.txt を空にする(max_count が更新されたときに呼ぶ)"""
    with open(SHIFT_PATH_FILE, "w", encoding="utf-8"):
        pass


def append_shift_path_file(state: "SearchState", first: bool = False) -> None:
    """shift_path.txt に shift_path を1行追加する"""
    with open(SHIFT_PATH_FILE, "a", encoding="utf-8") as f:
        if first:
            f.write(f"max_count:{state.max_count}\n")
        f.write(f"{state.shift_path}\n")


def shift_array(arr: npt.NDArray[np.bool_], k: int) -> npt.NDArray[np.bool_]:
    """
        k > 0 の時は右側にシフトする。
        k < 0 の時は左側にシフトする。
    """
    n: int = len(arr)
    if k >= 0:
        if k >= n:
            return np.zeros(n, dtype=arr.dtype)
        result = np.empty(n, dtype=arr.dtype)
        result[:k] = 0
        result[k:] = arr[: n - k]
        return result

    # k < 0: 左シフト
    shift: int = -k
    if shift >= n:
        return np.zeros(n, dtype=arr.dtype)
    result = np.empty(n, dtype=arr.dtype)
    result[n - shift:] = 0
    result[: n - shift] = arr[shift:]
    return result

def count_zero_mask(mask: npt.NDArray[np.bool_]) -> int:
    """真偽値マスク(各列が『これまでの全階層で0』かどうか)からcountを求める"""
    return int(np.count_nonzero(mask))

# 2,3,5,7,11,13,...,1579 の素数リスト(元コードのA2,A3,...,A1579に対応)
# 実体は config.py 側で管理する。
PRIMES: List[int] = cfg.PRIMES


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
    results : int
        max_count を達成した件数。max_count が更新されると1に初期化され、
        同じ max_count を達成するたびに1ずつ加算される。
        該当する shift_path 自体は SHIFT_PATH_FILE(shift_path.txt)に
        書き出される(max_count 更新時にファイルを空にし、count == max_count
        のたびに1行追加する)。
    count1 : int
        進捗表示用
    count2 : int
        進捗表示用
    primes : list[int]
        探索対象の素数リスト(先頭から順に1階層ずつ対応)。run_search で設定される。
    base_array : np.ndarray
        shape=(depth, COLS) の基準配列(build_base_array(primes[:depth]) で作成)。
        run_search で設定される。
    shift_path : list[int]
        現在の階層までの各素数に対するシフト値のリスト。search() 内で
        再帰の入り口で append、抜けるときに pop することで、呼び出しスタックと
        同期させる(値そのものは search() の引数として渡さない)。
    zero_mask : np.ndarray
        shape=(COLS,) の真偽値配列。「1つ上の階層までで全て0だった列」を表す。
        search() が子階層を呼び出す直前にその階層のマスクへ更新し、子の
        呼び出しがすべて終わったら元の値へ復元する(スタックのように使う)。
    """

    def __init__(self, limit: int, target: int, max_depth: int) -> None:
        self.limit: int = limit
        self.target: int = target
        self.max_depth: int = max_depth
        self.max_count: int = 0
        self.results: int = 0
        self.count1: int = 0
        self.count2: int = 0
        self.primes: Optional[List[int]] = None
        self.base_array: Optional[npt.NDArray[np.bool_]] = None
        self.shift_path: List[int] = []
        self.zero_mask: Optional[npt.NDArray[np.bool_]] = None

    def progress(self) -> None:
        """ 進捗表示 """
        self.count1 += 1
        if self.count1 % 100000 == 0:
            self.count1 = 0
            pos = self.count2 % 10
            print("-" * pos + "+" + "-" * (9 - pos))
            self.count2 = (self.count2 + 1) % 10        

def build_base_array(primes: List[int]) -> npt.NDArray[np.bool_]:
    """
    指定した素数リストからbase_arrayを作る。

    後段の処理(search)では「0かどうか」しか使わないため、値そのもの(素数p)
    ではなく bool(idx % p == 1) を格納する。int64(8byte/要素)ではなく
    bool(1byte/要素)にすることでメモリ転送量が1/8になり、shift_array や
    AND演算のコストが下がる。
    """
    return np.array([(IDX % p == 1) for p in primes])

def search(depth: int, state: SearchState) -> None:
    """
    再帰的に各階層のシフト値を探索する汎用関数。

    count_zero を毎回フルスキャンする代わりに、「これまでの階層すべてで0だった列」
    を表す真偽値マスク state.zero_mask を引き回し、1階層進めるたびに今回の行の
    0判定を AND するだけにすることで、count計算を O(depth) の毎回全スキャンから
    O(1) の増分更新に落としている(depth回の再帰全体で見ると O(depth^2) -> O(depth))。

    shift_path / primes / base_array / zero_mask はすべて引数ではなく
    state の属性として保持する。shift_path は再帰の入り口で append、
    抜けるときに pop することで呼び出しスタックと同期させ、zero_mask は
    子階層へ降りる直前に更新し、子の呼び出しがすべて終わったら元の値へ
    復元することで、従来「引数として都度渡していた」ものと同じスコープを
    実現している。

    Parameters
    ----------
    depth : int
        探索する階層数(= 使用する素数の個数)。可変。
    state : SearchState
        limit / target / max_depth といった探索パラメータ、探索中に更新される
        max_count / results、および primes / base_array / shift_path / zero_mask
        をまとめた状態オブジェクト。この1回の探索(run_search 呼び出し)の間
        だけ生存し、再帰全体で共有・更新される。
    """
    state.progress()
    level = len(state.shift_path) - 1  # 今回シフトを設定した階層(0-indexed)
    row_nonzero = shift_array(state.base_array[level], state.shift_path[level])
    mask = state.zero_mask & ~row_nonzero

    count = count_zero_mask(mask)
    logger.debug("depth=%d shift_path=%s count=%s", level + 1, state.shift_path, count)

    if count < state.limit or count < state.max_count:
        logger.debug("break shift_path=%s count=%d", state.shift_path, count)
        return  # この枝は打ち切り(子孫を探索しない)

    if level + 1 >= depth:
        """ 最深部ではcountがtargetを超えるのを無効とする """
        if depth == state.max_depth:
            if count > state.target:
                logger.debug("break shift_path=%s count=%d", state.shift_path, count)
                return

        if count > state.max_count:
            state.max_count = count
            logger.info("max_count=%d", state.max_count)
            state.results = 1
            clear_shift_path_file()
            append_shift_path_file(state, True)
            logger.debug("done shift_path=%s count=%d", state.shift_path, count)
        elif count == state.max_count:
            state.results += 1
            append_shift_path_file(state)
            logger.debug("done shift_path=%s count=%d", state.shift_path, count)
        return  # 最深階層に到達

    next_p = state.primes[level + 1]
    saved_mask = state.zero_mask
    state.zero_mask = mask
    for i in range(next_p):
        state.shift_path.append(i)
        search(depth, state)
        state.shift_path.pop()
    state.zero_mask = saved_mask


def run_search(
    primes: List[int],
    depth: int,
    limit: Optional[int] = None,
    target: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> SearchState:
    """
    primes[:depth] を使って深さ depth までの探索を実行するエントリポイント。

    limit / target / max_depth を省略した場合は config.py の値(cfg.LIMIT /
    cfg.TARGET / cfg.MAX_DEPTH)を使う。呼び出しごとに新しい SearchState を
    作成するため、同一プロセス内で run_search() を複数回呼んでも前回の
    max_count / results が次回の探索に混入することはない。

    Returns
    -------
    SearchState
        探索完了後の状態(state.results に max_count を達成した件数、
        state.max_count に達成した最大カウントが入っている)。
    """
    if depth > len(primes):
        raise ValueError(f"depth={depth} が primes の長さ({len(primes)})を超えています")

    state = SearchState(
        limit=cfg.LIMIT if limit is None else limit,
        target=cfg.TARGET if target is None else target,
        max_depth=cfg.MAX_DEPTH if max_depth is None else max_depth,
    )
    state.primes = primes
    state.base_array = build_base_array(primes[:depth])
    state.zero_mask = np.ones(COLS, dtype=bool)
    state.shift_path = []

    first_p = primes[0]
    for i in range(first_p):
        state.shift_path.append(i)
        search(depth, state)
        state.shift_path.pop()

    return state


if __name__ == "__main__":
    logger.info("HLSearch 開始 (log file: %s)", LOG_FILE)

    DEPTH = cfg.DEPTH  # 使用する素数の個数(可変)。元コードの calc2->calc3->calc5->calc7 相当
    state = run_search(PRIMES, DEPTH)

    logger.info("最大値: %d", state.max_count)
    logger.info("該当件数: %d", state.results)

    logger.info("HLSearch 終了")