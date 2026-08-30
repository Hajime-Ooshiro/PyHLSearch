# HLSearch

ハーディ・リトルウッドの第2予想に関する探索プログラム。

## 概要
- 2 から 1579 の素数を使い、各階層でのシフト候補を探索する。
- `State` クラスで反復DFSを実装し、再帰の制限やコールスタックを避ける。
- 事前計算したシフトテーブルで高速化し、探索条件に基づいて枝刈りを行う。
- 既定設定は `SearchConfig` で管理し、`generate_primes()` で素数列を自動生成する。
- 実装には `HLSearch.py`（NumPy版）、`HLSearch_Numba.py`（Numba版）、`HLSearch_bitpack.py`（純Python bit-packed版）がある。

## 実装一覧
- `HLSearch.py`: 既定の NumPy ベース実装。標準運用向け。
- `HLSearch_Numba.py`: Numba JIT / CUDA 対応版。高速度化を狙う。
- `HLSearch_bitpack.py`: NumPy を使わず `int` のビット演算で処理する軽量版。
- `HLSearch_Beam.py`: Beam 検索 + GPU/CPU 並列化の実験版。

## 中断・再開
`HLSearch.py` では探索の途中経過をテキスト checkpoint として保存でき、再開可能です。

```powershell
python HLSearch.py --depth 8 --limit 400 --max-depth 249 --target 447 --checkpoint resume.txt
python HLSearch.py --depth 8 --limit 400 --max-depth 249 --target 447 --resume resume.txt
```

- `--checkpoint`: 途中経過を `resume.txt` のようなファイルへ保存
- `--resume`: その checkpoint から探索を再開

## 動作環境
- Python 3.10 以上
- NumPy（`HLSearch.py` で必要）
- tqdm
- Numba（`HLSearch_Numba.py` を使う場合）

## インストール
```powershell
python -m pip install numpy tqdm numba
```

## 実行方法
### 基本実行（NumPy版）
```powershell
python HLSearch.py --depth 8 --limit 400 --max-depth 249 --target 447
```

### Numba版
```powershell
python HLSearch_Numba.py --numba --depth 8 --limit 400 --max-depth 249 --target 447
```

### Bit-packed版
```powershell
python HLSearch_bitpack.py --depth 8 --limit 400 --max-depth 249 --target 447
```

よく使うオプション:
- `-d, --depth`: 探索する深さ
- `-l, --limit`: 枝刈りの下限
- `--max-depth`: 最大深さ
- `-t, --target`: 最大深さ時の目標値
- `-p, --primes-count`: 先頭 N 個の素数だけ使用
- `--cols`: 列数
- `--output`: 結果の出力先ファイル
- `--checkpoint`: 途中経過を保存する checkpoint ファイル
- `--resume`: checkpoint から再開
- `--numba`: `HLSearch_Numba.py` で JIT 有効化
- `--cuda`: `HLSearch_Numba.py` で CUDA を優先使用

## 例: 小規模テスト
```powershell
python HLSearch.py --depth 3 --cols 50 --limit 0 --output shift_path.txt
python HLSearch.py --depth 3 --cols 50 --limit 0 --output shift_path.txt --checkpoint resume.txt
python HLSearch_Numba.py --numba --depth 3 --cols 50 --limit 0 --output shift_path.txt
python HLSearch_bitpack.py --depth 3 --cols 50 --limit 0 --output shift_path.txt
```

## テスト
```powershell
pytest -q
```

## 変更履歴
<details>
<summary><b>v1.0.5</b> (2026-08-30)</summary>

### 追加 (Added)
- `HLSearch_Numba.py` を追加（Numba/JIT ベース高速化）
- `HLSearch_bitpack.py` を追加（NumPy 非依存 bit-packed 実装）
- `HLSearch_Beam.py` で GPU/CPU ビーム探索を整理

### 変更 (Changed)
- README に複数実装の利用方法を追記
- 実行手順と依存関係を更新

</details>

<details>
<summary><b>v1.0.4</b> (2026-08-30)</summary>

### 変更 (Changed)
- `PRIMES` をハードコードから `generate_primes()` に変更
- `SearchConfig` に設定を集約
- `Config.py` を削除し、本体とテストを整理
- `FastSearcher` / `build_bit_tables` を役割の明確な `State` ベースへ整理

</details>

<details>
<summary><b>v1.0.3</b> (2026-08-27)</summary>

### 追加 (Added)
- 探索状態クラス `State` を使用
- 事前計算を導入
- `Config.py` を削除

</details>

<details>
<summary><b>v1.0.2</b> (2026-08-27)</summary>

### 追加 (Added)
- 進捗表示処理を追加
- ビームサーチを追加
- CUDA 向け強化を実施
- CLI 引数に対応

</details>

<details>
<summary><b>v1.0.1</b> (2026-08-23)</summary>

### 追加 (Added)
- 設定ファイル `Config.py` を追加

### 変更 (Changed)
- 再帰関数を使うよう変更

### 修正 (Fixed)
- プロジェクト名を修正

</details>

<details>
<summary><b>v1.0.0</b> (2026-08-21)</summary>

### 追加 (Added)
- 初版リリース

</details>
