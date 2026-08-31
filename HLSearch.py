"""HLSearch: 素数シフト探索プログラム。

このモジュールは、ハーディ・リトルウッドの第2予想に関する探索問題を
NumPy ベースの bitmask/shift table で高速化するための実装を提供する。

公開 API:
- generate_primes(limit): 指定上限までの素数を生成する
- SearchConfig: 探索設定をまとめた dataclass
- build_base_rows(primes): 素数ごとの基底行を生成する
- build_shift_table(primes): シフト候補テーブルを生成する
- State: 探索を実行する状態管理クラス
- parse_args(argv): CLI 引数を解析する

利用者はこのモジュールを直接 import して `State` を使うか、
`python HLSearch.py` として CLI から起動する。
"""

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def generate_primes(limit: int) -> list[int]:
    """limit 以下の素数を昇順で返す。

    Args:
        limit: 上限値。2 以上を指定する。

    Returns:
        limit 以下の素数を昇順に並べたリスト。
    """
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start:limit + 1:p] = b"\x00" * (((limit - start) // p) + 1)
    return [n for n in range(2, limit + 1) if sieve[n]]


@dataclass(frozen=True)
class SearchConfig:
    """探索処理に必要な設定をまとめた構成体。

    Attributes:
        primes: 探索対象の素数リスト。デフォルトでは 1579 以下の素数を生成する。
        depth: 深さとして使う素数の数。
        limit: 枝刈りの下限値。
        max_depth: 深さの上限。
        target: `depth == max_depth` のときの打ち切り目標値.
        cols: 列数。
        progress_mininterval: tqdm の最短更新間隔。
        postfix_update_interval: postfix 更新の頻度。
        shift_path_file: 出力ファイルパス.
    """
    primes: Sequence[int] = field(default_factory=lambda: generate_primes(1579))
    depth: int = 8
    limit: int = 447
    max_depth: int = 249
    target: int = 447
    cols: int = 3159
    progress_mininterval: float = 1.0
    postfix_update_interval: int = 10000
    shift_path_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_path.txt")

    def __post_init__(self) -> None:
        """設定値の整合性を早期に検証する(実行時ではなく構築時に失敗させる)。"""
        if self.cols <= 0:
            raise ValueError(f"cols は正の整数である必要があります: cols={self.cols}")
        if self.depth < 0:
            raise ValueError(f"depth は0以上である必要があります: depth={self.depth}")
        if self.depth > len(self.primes):
            raise ValueError(
                f"depth={self.depth} が primes の要素数({len(self.primes)})を超えています"
            )
        if self.limit < 0:
            raise ValueError(f"limit は0以上である必要があります: limit={self.limit}")
        if self.postfix_update_interval <= 0:
            raise ValueError(
                f"postfix_update_interval は正の整数である必要があります: "
                f"postfix_update_interval={self.postfix_update_interval}"
            )

DEFAULT_CONFIG = SearchConfig()
shift_path_file: str = DEFAULT_CONFIG.shift_path_file

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


COLS: int = DEFAULT_CONFIG.cols
LIMIT: int = DEFAULT_CONFIG.limit

DEPTH: int = DEFAULT_CONFIG.depth
MAX_DEPTH: int = DEFAULT_CONFIG.max_depth
TARGET: int = DEFAULT_CONFIG.target
PROGRESS_MININTERVAL: float = DEFAULT_CONFIG.progress_mininterval  # tqdm進捗表示の最短更新間隔(秒)
POSTFIX_UPDATE_INTERVAL: int = DEFAULT_CONFIG.postfix_update_interval

# 2,3,5,7,11,13,...,1579 の素数リスト
PRIMES: list[int] = generate_primes(1579)

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
 

def build_base_rows(primes: Sequence[int], cols: int = COLS) -> NDArray[np.bool_]:
    """指定した素数リストから各階層の基底行を生成する。

    各要素は `bool((idx % p) == 1)` を保持し、探索では「0かどうか」だけを
    判定する。`bool` 型にすることで 1 要素あたりのメモリ使用量を抑え、
    シフトや AND 演算を高速化する。

    Args:
        primes: 基底行を作る素数一覧。
        cols: 配列の列数。

    Returns:
        shape=(len(primes), cols) の bool 配列。
    """
    idx = np.arange(1, cols + 1)
    return np.array([(idx % p == 1) for p in primes])


def build_shift_table(primes: Sequence[int], cols: int = COLS) -> list[NDArray[np.bool_]]:
    """各レベルごとのシフト候補テーブルを事前生成する。

    これにより探索時に毎回 `shift_array()` と `~` 演算を行わず、
    事前に補集合を計算済みの配列をそのまま使える。

    Args:
        primes: 使用する素数のリスト。
        cols: 列数。

    Returns:
        `shift_table[level][shift]` が、level 段目におけるシフト値 `shift`
        に対応する補集合行を表す bool 配列。
    """
    base_rows = build_base_rows(primes, cols)
    shift_table: list[NDArray[np.bool_]] = []
    for level, p in enumerate(primes):
        row = base_rows[level]
        shifted_complement = np.empty((p, cols), dtype=bool)
        for k in range(p):
            shifted_complement[k] = ~shift_array(row, k)
        shift_table.append(shifted_complement)
    return shift_table

class State:
    """探索処理の状態を保持し、反復 DFS を実行する。

    This class owns the current search path (`key`), the active zero-mask,
    the pruning thresholds, and the best result state seen so far.

    Public API:
    - `search(depth)`: 指定深さまで DFS を実行する
    - `run(depth=None)`: 既定設定を使って探索を実行し、self を返す
    - `max_count`, `results`, `shifts`: 最良結果の集計
    """

    __slots__ = (
        "config",
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
        "checkpoint_path",
        "checkpoint_interval",
        "_stack",
    )

    def __init__(self, config: SearchConfig | Sequence[int], shift_table: list[NDArray[np.bool_]], limit: int | None = None, max_depth: int | None = None, target: int | None = None, checkpoint_path: str | os.PathLike[str] | None = None, checkpoint_interval: int = 1000) -> None:
        # SearchConfig 以外(生の primes 列)が渡された場合は、まず SearchConfig に
        # 正規化してしまう。これにより以降の属性代入を両ケースで共通化でき、
        # limit/max_depth/target の決定ロジックを二重に書かずに済む。
        if not isinstance(config, SearchConfig):
            config = SearchConfig(
                primes=config,
                depth=len(config),
                limit=DEFAULT_CONFIG.limit if limit is None else limit,
                max_depth=DEFAULT_CONFIG.max_depth if max_depth is None else max_depth,
                target=DEFAULT_CONFIG.target if target is None else target,
                cols=COLS,
            )
        self.config = config
        primes = config.primes
        self.limit = config.limit if limit is None else limit
        self.max_depth = config.max_depth if max_depth is None else max_depth
        self.target = config.target if target is None else target

        self.key: list[int] = []
        self.primes: Sequence[int] = primes
        self.shift_table: list[NDArray[np.bool_]] = shift_table
        self.zero_mask: NDArray[np.bool_] = np.ones(self.config.cols, dtype=bool)
        self.max_count: int = 0
        self.shifts: list[list[int]] = []
        self.results: int = 0
        self.start_time: float = time.time()
        self.node_count: int = 0
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_interval = checkpoint_interval
        self._stack: list[list] = []
        self.pbar = tqdm(
            desc="search",
            unit="node",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=self.config.progress_mininterval,
        )

    def _mask_to_int(self, mask: NDArray[np.bool_]) -> int:
        """bool配列をビット列とみなして多倍長整数に変換する。

        以前は `1 << np.arange(mask.size, dtype=np.int64)` で重み配列を作り
        `np.dot` していたが、mask.size が64を超えると int64 の範囲を超えて
        符号オーバーフローし、チェックポイントの内容が壊れるバグがあった。
        `np.packbits` でバイト列化してから Python の多倍長整数に変換すれば
        サイズに関わらず正しく、かつベクトル化されて高速。
        """
        if mask.size == 0:
            return 0
        packed = np.packbits(np.asarray(mask, dtype=bool))
        return int.from_bytes(packed.tobytes(), byteorder="big")

    def _int_to_mask(self, value: int, size: int) -> NDArray[np.bool_]:
        """`_mask_to_int` の逆変換。"""
        if size == 0:
            return np.zeros(0, dtype=bool)
        nbytes = (size + 7) // 8
        packed = np.frombuffer(value.to_bytes(nbytes, byteorder="big"), dtype=np.uint8)
        return np.unpackbits(packed)[:size].astype(bool)

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stack_payload = []
        for level, base_mask, next_idx, next_p in self._stack:
            stack_payload.append({
                "level": int(level),
                # 多倍長整数はJSONの数値としてそのままシリアライズ可能。
                "base_mask": self._mask_to_int(base_mask),
                "next_idx": int(next_idx),
                "next_p": int(next_p),
            })

        saved = {
            "version": 1,
            "key": list(self.key),
            "zero_mask": self._mask_to_int(self.zero_mask),
            "max_count": self.max_count,
            "results": self.results,
            "shifts": self.shifts,
            "node_count": self.node_count,
            "stack": stack_payload,
        }
        # 途中でクラッシュしても既存のチェックポイントを壊さないよう、
        # 一時ファイルに書いてから rename でatomicに置き換える。
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _load_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return
        try:
            saved = json.loads(text)
        except json.JSONDecodeError as exc:
            # 旧バージョン(key=value形式のテキスト)のチェックポイントを
            # 読み込もうとした場合に、原因不明のJSONエラーではなく
            # 何が問題かが分かるメッセージにする。
            raise ValueError(
                f"{path} はJSON形式のチェックポイントとして読み込めません。"
                "旧バージョン(key=value形式)のチェックポイントには対応していません。"
                "新しい設定で最初から探索をやり直してください。"
            ) from exc

        if saved.get("version") != 1:
            raise ValueError(f"unsupported checkpoint version: {saved.get('version')}")

        self.key = list(saved.get("key", []))
        self.zero_mask = self._int_to_mask(int(saved.get("zero_mask", 0)), self.config.cols)
        self.max_count = int(saved.get("max_count", 0))
        self.results = int(saved.get("results", 0))
        self.node_count = int(saved.get("node_count", 0))
        self.shifts = list(saved.get("shifts", []))
        raw_stack = saved.get("stack", [])
        self._stack = [
            [
                int(entry["level"]),
                self._int_to_mask(int(entry["base_mask"]), self.config.cols),
                int(entry["next_idx"]),
                int(entry["next_p"]),
            ]
            for entry in raw_stack
        ]

    def report_progress(self, force: bool = False) -> None:
        """
        tqdmの進捗バーを更新する(ログファイルには出さない)。

        呼び出しが多い(再帰の全ノードで呼ばれる)ため、実際の画面再描画は
        tqdm側が mininterval 秒に一度だけに間引いてくれる。force=True の
        ときは set_postfix に refresh=True を渡し、間引かずに必ず再描画する
        (探索開始・終了時など)。
        """
        if self.node_count % self.config.postfix_update_interval == 0:
            self.pbar.update(1)
            self.pbar.set_postfix(
                best=self.max_count,
                hits=self.results,
                depth=len(self.key),
                key=list(self.key),
                refresh=force,
            )
        if self.checkpoint_path is not None and self.node_count % self.checkpoint_interval == 0:
            self._save_checkpoint()

    def search(self, depth: int) -> None:
        """
        各階層のシフト値を探索する(反復版・スタックによる明示的DFS)。

        元の実装は「1階層シフトを決める→自分自身を再帰呼び出しして
        次の階層を決める」という再帰関数だったが、depth(ひいては
        再帰の深さ)が大きくなると Python の再帰上限(sys.setrecursionlimit)
        や関数呼び出しオーバーヘッドが問題になりうる。
        ここでは再帰呼び出しの代わりに、階層ごとの「ループの途中状態」を
        自前のスタック `self._stack` に積んで管理することで、同じ探索順序・
        同じ結果を非再帰(反復)で実現する。

        `self._stack` はインスタンス属性として持たせているため、
        `_save_checkpoint`/`_load_checkpoint` からもそのまま読み書きできる。
        そのため、探索の全ノードごとに別のコピーへ同期する必要はなく、
        チェックポイントを実際に保存するタイミング(`checkpoint_interval`
        ノードに1回)でだけ JSON へシリアライズすればよい。

        スタックの各要素は [level, base_mask, next_idx, next_p] の
        4要素からなる可変(list)フレーム:
            level      : このループで値を決める階層(0-indexed)。
                         元の再帰版での level = len(key) - 1 に対応。
            base_mask  : この階層のどの枝を試す場合でも共通して使う
                         「親までの zero_mask」。元の再帰版での
                         prev_mask(= self.zero_mask の呼び出し前の値)
                         に対応する。
            next_idx   : 次に試す `range(next_p)` 上のインデックス。
                         元の再帰版の re-entrant な `for i in range(...)`
                         ループの「途中状態」をこれで表現する。
            next_p     : この階層で使う素数(= primes[level])。
                         next_idx の終了条件として保持している。

        1つの節点(= key の1要素)を「探索し尽くして親に戻る」タイミングは、
        自分の子階層で next_idx が next_p に達した瞬間として検出し、
        そこで元の再帰版の「self.zero_mask = prev_mask; return」に相当する
        後始末(zero_maskの復元・keyのpop)を行う。

        Parameters
        ----------
        depth : int
            探索する階層数(= 使用する素数の個数)。可変。
        """
        key = self.key
        stack = self._stack
        if not stack:
            stack.append([0, self.zero_mask.copy(), 0, self.primes[0]])

        while stack:
            frame = stack[-1]
            level, base_mask, next_idx, next_p = frame
            if next_idx >= next_p:
                finished_base_mask = stack.pop()[1]
                if stack:
                    key.pop()
                    self.zero_mask = stack[-1][1]
                else:
                    self.zero_mask = finished_base_mask  # 最上位まで戻り切った
                continue

            i = next_idx
            frame[2] = next_idx + 1  # 同じフレームをその場で更新(コピー不要)

            key.append(i)
            self.node_count += 1

            # report_progress()(チェックポイント保存を含む)は、
            # key と stack の対応関係が「len(key) == len(stack) - 1」に
            # 揃っているタイミングでしか呼んではいけない。
            # 枝刈り/葉ノード判定/子階層プッシュのどの分岐を取るかが
            # 確定する前(= key.append 直後)に呼んでしまうと、
            # 「i を試している最中」という中途半端な状態のまま
            # チェックポイントに保存されてしまい、再開時に stack 側の
            # next_idx だけが1つ先に進んだ不整合な状態から再開してしまう
            # (key が本来ポップされるべきだったエントリを残したまま
            # 際限なく伸び続けるバグになる)。
            # そのため、分岐の結果が確定した直後(continue する直前、
            # または子階層をpushした直後)に必ず1回だけ呼ぶよう
            # try/finally で保証する。
            try:
                row_complement = self.shift_table[level][i]  # ~row_nonzero(NOT演算済み、事前作成済み)
                node_mask = base_mask & row_complement
                count = int(np.count_nonzero(node_mask))

                if count < max(self.limit, self.max_count):
                    key.pop()
                    continue

                if level + 1 >= depth:
                    if not (depth == self.max_depth and count > self.target):
                        if count > self.max_count:
                            self.max_count = count
                            self.results = 1
                            self.shifts.clear()
                            self.shifts.append(list(key))
                            message = f"done key={list(key)} count={count}"
                            self.pbar.write(message)
                            logger.info(message)  # ログファイルにも残す(pbar.writeだけだと画面にしか出ない)
                        elif count == self.max_count:
                            self.results += 1
                            self.shifts.append(list(key))

                    key.pop()
                    continue

                self.zero_mask = node_mask
                next_p_child = self.primes[level + 1]
                stack.append([level + 1, node_mask, 0, next_p_child])
            finally:
                self.report_progress()

    def run(self, depth: int | None = None, resume_from: str | os.PathLike[str] | None = None) -> "State":
        """primes[:depth] を使って深さ depth までの探索を実行するエントリポイント"""
        depth_to_use = self.config.depth if depth is None else depth
        if depth_to_use > len(self.primes):
            raise ValueError(f"depth={depth_to_use} が使用可能な素数の個数({len(self.primes)})を超えています")
        if resume_from is not None:
            self._load_checkpoint(resume_from)
        try:
            self.search(depth_to_use)
            self.report_progress(force=True)
        finally:
            self.pbar.close()
            if self.checkpoint_path is not None:
                self._save_checkpoint()

        return self


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI 引数を解釈して探索設定を返す。

    Args:
        argv: 引数リスト。None の場合は `sys.argv[1:]` を使う。

    Returns:
        argparse.Namespace 形式の設定。
    """
    parser = argparse.ArgumentParser(
        description="HLSearch: 素数シフト探索プログラム",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--depth", type=int, default=DEPTH,
                        help="探索する階層数(使用する素数の個数)。primesの長さ以下である必要がある。")
    parser.add_argument("-l", "--limit", type=int, default=LIMIT,
                        help="打ち切りに使うcountの下限値。これ未満の枝は探索しない。")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                        help="depthがこの値と一致するとき、--targetによる追加打ち切りを有効にする。")
    parser.add_argument("-t", "--target", type=int, default=TARGET,
                        help="depth == max-depth のとき、countがこの値を超えたら結果を採用せず打ち切る。")
    parser.add_argument("-p", "--primes-count", type=int, default=None, metavar="N",
                        help="PRIMESの先頭N個だけを使う(未指定なら全て使用)。")
    parser.add_argument("--cols", type=int, default=COLS,
                        help="列数(=探索対象の長さ)。")
    parser.add_argument("--output", type=str, default=shift_path_file,
                        help="最適シフトパスの出力先ファイル。")
    parser.add_argument("--mininterval", type=float, default=PROGRESS_MININTERVAL,
                        help="tqdm進捗表示の最短更新間隔(秒)。")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="コンソールに出すログレベル(ログファイルは常にDEBUG)。")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="途中経過をJSON形式で保存するパス。再開時は --resume を使う。")
    parser.add_argument("--resume", type=str, default=None,
                        help="保存済み checkpoint から探索を再開する。")
    return parser.parse_args(argv)


if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0]).lower()
    argv = sys.argv[1:] if script_name not in {"pytest", "py.test"} else []
    args = parse_args(argv)

    base = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = setup_logging(base, console_level=args.log_level)

    depth = args.depth
    limit = args.limit
    max_depth = args.max_depth
    target = args.target
    cols = args.cols
    primes = PRIMES if args.primes_count is None else PRIMES[: args.primes_count]
    output_path = args.output

    if depth > len(primes):
        raise ValueError(f"depth={depth} が使用可能な素数の個数({len(primes)})を超えています")

    config = SearchConfig(
        primes=primes,
        depth=depth,
        limit=limit,
        max_depth=max_depth,
        target=target,
        cols=cols,
        progress_mininterval=args.mininterval,
        postfix_update_interval=POSTFIX_UPDATE_INTERVAL,
        shift_path_file=output_path,
    )

    logger.info("HLSearch 開始 (log file: %s)", LOG_PATH)
    logger.info("設定: depth=%d limit=%d max_depth=%d target=%d primes_count=%d", depth, limit, max_depth, target, len(primes))

    shift_table = build_shift_table(primes[:depth], cols)
    state = State(config, shift_table, checkpoint_path=args.checkpoint, checkpoint_interval=max(1, min(10000, max(10, depth * 100))))
    result_state = state.run(depth, resume_from=args.resume)

    logger.info("最大値: %d", result_state.max_count)
    logger.info("該当件数: %d", result_state.results)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"max_count:{result_state.max_count}\n")
        f.write(f"results:{result_state.results}\n")
        for shift in result_state.shifts:
            f.write(f"{shift}\n")

    logger.info("HLSearch 終了")