# HLSearch

ハーディ・リトルウッド第 2 予想に関連する素数シフト探索プログラムです。

## 概要

- 2 から 1579 までの素数を候補として、各階層ごとのシフトを探索する
- `State` は反復 DFS を使って再帰制限やコールスタックの問題を避ける
- シフトテーブルと探索マスクを `uint64` ワードへビットパックし、AND と popcount をワード単位で処理する
- 各枝で `max_count` 未満の候補を早期に打ち切る
- `results` は最終最大値を持つパス数、`shifts` はその最大値パスの集合を保持する
- `--workers` を指定すると、最上位シフトを複数プロセスへ分配し、親プロセスで結果を統合する
- 既定設定は `SearchConfig` で管理し、`generate_primes()` で素数列を自動生成する
- 標準実装は `HLSearch.py` で、実験的な変種は `bk/` に保管されている

## リポジトリ構成

- `HLSearch.py`: 標準実装
- `tests/`: 検証用テスト群
- `bk/`: 過去の実験コードや比較用コード
- `.github/copilot-instructions.md`: 開発方針と設計メモ

## 動作環境

- Python 3.10 以上
- NumPy 2.0 以上
- tqdm
- CUDA 利用時は CuPy と NVIDIA CUDA ドライバ（任意）

## インストール

```powershell
python -m pip install "numpy>=2.0" tqdm
```

CUDA を使う場合は、対応する CuPy も入れてください。

```powershell
python -m pip install cupy-cuda12x
```

既定では NumPy の CPU popcount を使います。CuPy と CUDA デバイスが使える環境では `--cuda` を指定すると GPU 側で popcount を試せます。利用できない場合は自動的に CPU にフォールバックします。

## パフォーマンス

探索用のシフトテーブルとマスクは `uint64` にビットパックされます。列数を `cols` とすると、各マスクの長さは `ceil(cols / 64)` ワードです。Boolean 配列と比べてメモリ使用量はおおむね 1/8 に抑えられます。

`--cuda` は popcount のみを GPU 上で処理します。転送と同期のコストが大きいため、一般的には CPU のほうが速い場合があります。環境と探索サイズを見て使い分けてください。

## 基本的な実行方法

```powershell
python HLSearch.py --depth 8
```

### 小規模テスト

```powershell
python HLSearch.py --depth 3 --cols 50 --output shift_path.txt
```

### 中断・再開

```powershell
python HLSearch.py --depth 8 --checkpoint resume.json
python HLSearch.py --depth 8 --resume resume.json
```

- `--checkpoint`: 途中経過を JSON 形式で保存する
- `--resume`: 保存済み checkpoint から再開する
- checkpoint の形式バージョンは `version: 4` のみをサポートする。古い `key=value` 形式や、ビットパック化前の JSON checkpoint は読み込めないため、現在の設定で新規に保存したファイルを使うこと
- 並列探索の checkpoint は `version: 1` の別形式で、完了済み最上位枝を記録する。再開時は未完了枝を実行する

## 結果出力

`--output`（既定: `shift_path.txt`）には次の形式で結果を書き出します。

```text
max_count:<最終最大値>
results:<最終最大値パス数>
[<shift 0>, <shift 1>, ...]
```

シフトパスは最終最大値パスの集合です。

## 代表的なオプション

- `-d, --depth`: 探索する深さ
- `--cols`: 列数
- `--cuda`: CuPy/CUDA による popcount を有効化
- `--workers`: 最上位シフトを分担するプロセス数
- `--output`: 結果出力先ファイル
- `--mininterval`: 進捗表示の最短更新間隔（秒）
- `--log-level`: コンソールログレベル
- `--checkpoint`: 定期保存する checkpoint ファイル
- `--resume`: checkpoint から再開

```powershell
python HLSearch.py --depth 4 --cols 500 --output shift_path.txt
```

## テスト

```powershell
pytest -q
```

詳細な実装方針、デバッグ方法、チェックポイント設計については `.github/copilot-instructions.md` を参照してください。

<details>

<summary><b>v1.0.7</b> (2026-08-31)</summary>

### 変更 (Changed)

- `tests/` 配下にテストを整理し、関連テストが配置場所に依存しないよう統一
- `report_progress()` の更新頻度を `postfix_update_interval` に揃えて、進捗表示が一貫するよう修正

### 修正 (Fixed)

- 各実装で不足していた `SearchConfig.__post_init__` を整備し、設定の検証が一貫するよう修正

### 追加 (Added)

</details>

<details>

<summary><b>v1.0.6</b> (2026-08-30)</summary>

### 変更 (Changed)

- checkpoint を JSON 形式へ変更し、再開の信頼性を向上

### 修正 (Fixed)

- 大きな `cols` で checkpoint のビットマスクが overflow して再開時に壊れる問題を修正
- `--log-level` 指定時にコンソールログが変わらない不具合を修正

</details>

<details>

<summary><b>v1.0.5</b> (2026-08-30)</summary>

- 主要探索ロジックを `HLSearch.py` に整理
- 旧来の実験実装を `bk/` に移管

### 変更 (Changed)

- README と実行手順を現行構成に合わせて更新
- 依存関係と利用例を整理

</details>

<summary><b>v1.0.4</b> (2026-08-30)</summary>

### 変更 (Changed)

- `PRIMES` を `generate_primes()` ベースへ整理
- `SearchConfig` に設定を集約
- 実験コードを `bk/` に整理

</details>
<details>

<summary><b>v1.0.3</b> (2026-08-27)</summary>

### 追加 (Added)

- `State` クラスによる探索状態管理を導入
- 事前計算テーブルを活用する設計に整理

</details>

<details>

<summary><b>v1.0.2</b> (2026-08-27)</summary>
### 追加 (Added)

- 進捗表示を追加
- CLI 引数による実行を追加

</details>

<details>
<summary><b>v1.0.1</b> (2026-08-23)</summary>
### 変更 (Changed)

- 再帰探索から反復型探索へ整理

</details>

<details>

<summary><b>v1.0.0</b> (2026-08-21)</summary>

### 追加 (Added)

- 初版リリース

</details>
