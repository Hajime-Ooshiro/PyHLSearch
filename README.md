# HLSearch

ハーディ・リトルウッドの第2予想に関する探索プログラム。

## 概要
- 2 から 1579 の素数を使い、各階層でのシフト候補を探索する。
- `State` クラスで反復DFSを実装し、再帰の制限やコールスタックを避ける。
- 事前計算したシフトテーブルで高速化し、探索条件に基づいて枝刈りを行う。
- 既定設定は `SearchConfig` で管理し、`generate_primes()` で素数列を自動生成する。

## 動作環境
- Python 3.10 以上
- numpy
- tqdm

## インストール
```powershell
python -m pip install numpy tqdm
```

## 実行方法
```powershell
python HLSearch.py --depth 8 --limit 400 --max-depth 249 --target 447
```

よく使うオプション:
- `-d, --depth`: 探索する深さ
- `-l, --limit`: 枝刈りの下限
- `--max-depth`: 最大深さ
- `-t, --target`: 最大深さ時の目標値
- `-p, --primes-count`: 先頭 N 個の素数だけ使用
- `--cols`: 列数
- `--output`: 結果の出力先ファイル

## 例: 小規模テスト
```powershell
python HLSearch.py --depth 3 --cols 50 --limit 0 --output shift_path.txt
```

## テスト
```powershell
pytest -q
```

## 変更履歴
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
