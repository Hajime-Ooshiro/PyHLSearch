# HLSearch.py (numpy高速化版)
import argparse
import logging
import logging.handlers
import os
import sys
import time
from collections.abc import Iterator, Sequence
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# --- logging設定 ---
logger = logging.getLogger(__name__)

def setup_logging(base_dir: str | os.PathLike[str], console_level: str="INFO") -> str:
    """コンソールとファイルの両方にログを出力するよう設定する。"""
    log_path = os.path.join(base_dir, "HLSearch.log")

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 二重登録防止(再実行・再インポート対策)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"ログファイルを作成しました: {log_path}")
    return log_path


COLS: int = 3159
IDX: NDArray[np.int64] = np.arange(1, COLS + 1)  # range(1, 3160) 
LIMIT: int = 400
DEPTH: int = 8
MAX_DEPTH: int = 249
TARGET: int = 447
PROGRESS_MININTERVAL: float = 1.0  # tqdm進捗表示の最短更新間隔(秒)
POSTFIX_UPDATE_INTERVAL: int = 10000

# 2,3,5,7,11,13,...,1579 の素数リスト
PRIMES: list[int] = [
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
ROWS: int = len(PRIMES)

def shift_array(arr: NDArray[np.bool_], k: int) -> NDArray[np.bool_]:
    """
    配列を右にk個シフトする(numpy版)。先頭k要素は0埋めし、末尾のk要素は捨てる。
    """
    n = len(arr)
    if k <= 0:
        return arr.copy()
    if k >= n:
        return np.zeros(n, dtype=arr.dtype)
    result = np.empty(n, dtype=arr.dtype)
    result[:k] = 0
    result[k:] = arr[: n - k]
    return result
 

def build_base_rows(primes: Sequence[int]) -> NDArray[np.bool_]:
    """
    指定した素数リストからbase_rows配列を作る。

    後段の処理(search)では「0かどうか」しか使わないため、値そのもの(素数p)
    ではなく bool(idx % p == 1) を格納する。int64(8byte/要素)ではなく
    bool(1byte/要素)にすることでメモリ転送量が1/8になり、shift_array や
    AND演算のコストが下がる。
    """
    return np.array([(IDX % p == 1) for p in primes])


def build_shift_table(primes: Sequence[int]) -> list[NDArray[np.bool_]]:
    """
    各階層(prime)ごとに、そのレベルで取り得る全てのシフト値
    k = 0, 1, ..., p-1 に対応する shift_array(row, k) の「補集合」
    (= ~shift_array(row, k))をあらかじめ計算してテーブル化する。

    search() では毎ノードごとに row_nonzero(=そのシフト値の行)に対して
    NOT演算を行い base_mask とANDを取っていたが、shift_table の中身は
    探索中不変なので、NOT演算はここで前もって1回だけ行っておけば十分。
    こうすることで search() のホットループでの numpy 呼び出し回数を
    1回減らせる(実測で約1.5倍高速化)。

    Parameters
    ----------
    primes : list[int]
        探索対象の素数リスト(depth分だけ、先頭から使う)。

    Returns
    -------
    list[np.ndarray]
        shift_table[level] は shape=(primes[level], COLS) の bool 配列。
        shift_table[level][k] が「level段目の基準行を k だけ右シフトした行の
        補集合」(= ~row_nonzero、search で言う node_mask を求めるのに
        そのまま base_mask とANDすればよい形)に対応する。
    """
    base_rows = build_base_rows(primes)
    shift_table: list[NDArray[np.bool_]] = []
    for level, p in enumerate(primes):
        row = base_rows[level]
        shifted_complement = np.empty((p, COLS), dtype=bool)
        for k in range(p):
            shifted_complement[k] = ~shift_array(row, k)
        shift_table.append(shifted_complement)
    return shift_table

class State:
    """
    search()/run() が探索全体(スタックによる反復探索)で引き回す状態をまとめたクラス。

    Attributes
    ----------
    key : list[int]
        現在の階層までの各素数に対するシフト値のリスト。
        呼び出し側が append/pop で管理する。
    primes : list[int]
        探索対象の素数リスト(先頭から順に1階層ずつ対応)。
    shift_table : list[np.ndarray]
        build_shift_table(primes[:depth]) で事前作成したシフト済み配列テーブル
        (あらかじめ ~(NOT) を取った補集合の形で格納されている)。
        shift_table[level][key[level]] が search() で必要な行(node_maskを
        求めるため base_mask と直接ANDすればよい形)を直接与えるため、
        探索中に shift_array や ~(NOT) 演算を呼ぶ必要がない。
    zero_mask : np.ndarray
        shape=(COLS,) の真偽値配列。「現在の階層までで全て0だった列」を表す。
        search() 内で1階層進むたびに更新され、その階層の探索が終わったら
        呼び出し前の値に戻される(backtrack)。
    limit : int
        count がこれより小さい場合はトラックバック
    max_depth : int
        深さの最大値
    target : int
        countの目標値
    max_count : int
        これまでに見つかった最大の count。
    shifts : list[list[int]]
        max_count を達成した key のリスト。
    results : int
        max_count を達成した組み合わせの件数。
    start_time : float
        探索開始時刻(time.time())。経過時間の表示に使う。
    node_count : int
        search() が呼ばれた回数(探索したノード数)。進捗表示に使う。
    pbar : tqdm
        進捗表示用の tqdm インスタンス。ノード数のカウントと
        best/hits/depth/key の postfix 表示に使う。
    """

    __slots__ = (
        "key",
        "primes",
        "shift_table",
        "zero_mask",
        "limit",
        "max_depth",
        "target",
        "max_count",
        "shifts",
        "results",
        "start_time",
        "node_count",
        "pbar",
    )

    def __init__(self, primes: Sequence[int], shift_table: list[NDArray[np.bool_]], limit: int=LIMIT, max_depth: int=MAX_DEPTH, target: int=TARGET) -> None:
        self.key: list[int] = []
        self.primes: Sequence[int] = primes
        self.shift_table: list[NDArray[np.bool_]] = shift_table
        self.zero_mask: NDArray[np.bool_] = np.ones(COLS, dtype=bool)
        self.limit: int = limit
        self.max_depth: int = max_depth
        self.target: int = target
        self.max_count: int = 0
        self.shifts: list[list[int]] = []
        self.results: int = 0
        self.start_time: float = time.time()
        self.node_count: int = 0
        self.pbar = tqdm(
            desc="search",
            unit="node",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=PROGRESS_MININTERVAL,
        )

    def report_progress(self, force: bool = False) -> None:
        """
        tqdmの進捗バーを更新する(ログファイルには出さない)。

        呼び出しが多い(再帰の全ノードで呼ばれる)ため、実際の画面再描画は
        tqdm側が mininterval 秒に一度だけに間引いてくれる。force=True の
        ときは set_postfix に refresh=True を渡し、間引かずに必ず再描画する
        (探索開始・終了時など)。
        """
        self.pbar.update(1)
        if self.node_count % POSTFIX_UPDATE_INTERVAL == 0:
            self.pbar.set_postfix(
                best=self.max_count,
                hits=self.results,
                depth=len(self.key),
                key=list(self.key),
                refresh=force,
            )

    def search(self, depth: int) -> None:
        """
        各階層のシフト値を探索する(反復版・スタックによる明示的DFS)。

        元の実装は「1階層シフトを決める→自分自身を再帰呼び出しして
        次の階層を決める」という再帰関数だったが、depth(ひいては
        再帰の深さ)が大きくなると Python の再帰上限(sys.setrecursionlimit)
        や関数呼び出しオーバーヘッドが問題になりうる。
        ここでは再帰呼び出しの代わりに、階層ごとの「ループの途中状態」を
        自前のスタックに積んで管理することで、同じ探索順序・同じ結果を
        非再帰(反復)で実現する。

        スタックの各要素は (level, base_mask, iterator) の3つ組:
            level      : このループで値を決める階層(0-indexed)。
                         元の再帰版での level = len(key) - 1 に対応。
            base_mask  : この階層のどの枝を試す場合でも共通して使う
                         「親までの zero_mask」。元の再帰版での
                         prev_mask(= self.zero_mask の呼び出し前の値)
                         に対応する。
            iterator   : range(primes[level]) の残り候補値を返す
                         イテレータ。元の re-entrant な `for i in range(...)`
                         ループの「途中状態」をこれで表現する。

        1つの節点(= key の1要素)を「探索し尽くして親に戻る」タイミングは、
        自分の子階層のイテレータが尽きた(StopIteration)瞬間として検出し、
        そこで元の再帰版の「self.zero_mask = prev_mask; return」に相当する
        後始末(zero_maskの復元・keyのpop)を行う。

        Parameters
        ----------
        depth : int
            探索する階層数(= 使用する素数の個数)。可変。
        """
        key = self.key
        stack: list[tuple[int, NDArray[np.bool_], Iterator[int]]] = [
            (0, self.zero_mask, iter(range(self.primes[0])))
        ]

        while stack:
            level, base_mask, it = stack[-1]
            try:
                i = next(it)
            except StopIteration:
                # この階層で試せる値を使い切った → 親の階層へbacktrack
                finished_base_mask = stack.pop()[1]
                if stack:
                    key.pop()
                    self.zero_mask = stack[-1][1]
                else:
                    self.zero_mask = finished_base_mask  # 最上位まで戻り切った
                continue

            key.append(i)
            self.node_count += 1
            self.report_progress()

            row_complement = self.shift_table[level][i]  # ~row_nonzero(NOT演算済み、事前作成済み)
            node_mask = base_mask & row_complement
            count = int(np.count_nonzero(node_mask))
            # logger.debug("depth=%d key=%s count=%s", level + 1, key, count)

            if count < max(self.limit, self.max_count):
                # logger.debug(("break", list(key), count))
                key.pop()  # この枝は打ち切り(子孫を探索しない)、次のiへ
                continue

            if level + 1 >= depth:
                if not (depth == self.max_depth and count > self.target):
                    if count > self.max_count:
                        self.max_count = count
                        self.results = 1
                        self.shifts.clear()
                        self.shifts.append(list(key))
                        self.pbar.write(f"done key={list(key)} count={count}")
                        # logger.debug(("done", list(key), count))
                    elif count == self.max_count:
                        self.results += 1
                        self.shifts.append(list(key))
                        # logger.debug(("done", list(key), count))

                key.pop()  # 最深階層に到達、次のiへ
                continue

            # さらに深く探索: 子階層のループをスタックに積んで先に進む
            self.zero_mask = node_mask
            next_p = self.primes[level + 1]
            stack.append((level + 1, node_mask, iter(range(next_p))))

    def run(self, depth: int) -> "State":
        """primes[:depth] を使って深さ depth までの探索を実行するエントリポイント"""
        if depth > len(self.primes):
            raise ValueError("depth={depth} が使用可能な素数の個数({len(self.primes)})を超えています")            

        try:
            self.search(depth)
            self.report_progress(force=True)
        finally:
            self.pbar.close()

        return self


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を解析する。

    未指定の項目はモジュール冒頭で定義済みのデフォルト値
    (DEPTH / LIMIT / MAX_DEPTH / TARGET / PROGRESS_MININTERVAL)を使う。
    """
    parser = argparse.ArgumentParser(
        description="HLSearch: 素数シフト探索プログラム",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=DEPTH,
        help="探索する階層数(使用する素数の個数)。primesの長さ以下である必要がある。",
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=LIMIT,
        help="打ち切りに使うcountの下限値。これ未満の枝は探索しない。",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help="depthがこの値と一致するとき、--targetによる追加打ち切りを有効にする。",
    )
    parser.add_argument(
        "-t", "--target",
        type=int,
        default=TARGET,
        help="depth == max-depth のとき、countがこの値を超えたら結果を採用せず打ち切る。",
    )
    parser.add_argument(
        "-p", "--primes-count",
        type=int,
        default=None,
        metavar="N",
        help="PRIMESの先頭N個だけを使う(未指定なら全て使用)。",
    )
    parser.add_argument(
        "--mininterval",
        type=float,
        default=PROGRESS_MININTERVAL,
        help="tqdm進捗表示の最短更新間隔(秒)。",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="コンソールに出すログレベル(ログファイルは常にDEBUG)。",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = setup_logging(base)

    # モジュールグローバル(search()が直接参照している)をCLI引数で上書き
    depth = args.depth
    PROGRESS_MININTERVAL = args.mininterval

    primes = PRIMES if args.primes_count is None else PRIMES[: args.primes_count]

    if depth > len(primes):
        raise ValueError(f"depth={depth} が使用可能な素数の個数({len(primes)})を超えています")

    logger.info("HLSearch 開始 (log file: %s)", LOG_PATH)
    logger.info(
        "設定: depth=%d limit=%d max_depth=%d target=%d primes_count=%d",
        depth, args.limit, args.max_depth, args.target, len(primes),
    )

    shift_table = build_shift_table(primes[:depth])
    state = State(primes, shift_table, args.limit, args.max_depth, args.target)
    result_state = state.run(depth)

    logger.info("最大値: %d",result_state.max_count)
    logger.info("該当件数: %d", result_state.results)
    SHIFT_PATH_FILE = os.path.join(base, "shift_path.txt")
    with open(SHIFT_PATH_FILE, "w", encoding="utf-8") as f:
        f.write(f"max_count:{result_state.max_count}\n")
        f.write(f"results:{result_state.results}\n")
        for shift in result_state.shifts:
            f.write(f"{shift}\n")

    logger.info("HLSearch 終了")