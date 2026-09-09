# HLSearch

ハーディ・リトルウッドの第2予想に関する探索プログラムです。

## 概要

- 2 から 1579 までの素数を対象に、各階層でのシフト候補を探索する。
- `search()` は `SearchConfig.primes`（PRIMES）を基準に、各階層のシフト候補を最大値から 0 へ降順で探索する。
- `State` クラスで反復 DFS を実装し、再帰制限やコールスタックの問題を避ける。
- 事前計算したシフトテーブルを使って探索を高速化し、必要に応じて枝刈りを行う。
- 既定設定は `SearchConfig` で管理し、`generate_primes()` で素数列を自動生成する。
- 現在のリポジトリの標準実装は `HLSearch.py` であり、他の変種は過去の実験コードとして `bk/` に保管されている。

## リポジトリ構成

- `HLSearch.py`: 現在の標準実装。NumPy ベースの探索エンジン。
- `tests/`: 統合テスト、チェックポイント再開テストなど。
- `bk/`: 過去の実験実装や比較用コード。
- `.github/copilot-instructions.md`: Copilot 向けの開発ガイド。

## 動作環境

- Python 3.10 以上
- NumPy
- tqdm
- CUDA 利用時は CuPy と NVIDIA CUDA ドライバ（任意）

## インストール

```powershell
python -m pip install numpy tqdm
```

CUDA を利用する場合は、環境に合った CuPy パッケージもインストールしてください。

```powershell
python -m pip install cupy-cuda12x
```

既定では NumPy による CPU popcount を使用します。CUDA デバイスと CuPy が利用可能な環境で CUDA popcount を試す場合は、`--cuda` を指定してください。CUDA を利用できない場合は NumPy にフォールバックします。

## 基本的な実行方法

```powershell
python HLSearch.py --depth 8 --max-depth 249 --target 447
```

### 小規模テスト

```powershell
python HLSearch.py --depth 3 --cols 50 --output shift_path.txt
```

### 中断・再開

```powershell
python HLSearch.py --depth 8 --max-depth 249 --target 447 --checkpoint resume.json
python HLSearch.py --depth 8 --max-depth 249 --target 447 --resume resume.json
```

- `--checkpoint`: 途中経過を JSON 形式で保存する
- `--resume`: 保存済み checkpoint から探索を再開する
- 旧形式の `key=value` checkpoint と、昇順探索時に作成した JSON checkpoint は読み込まないため、現在の設定で新規に保存したファイルを使うこと

## 代表的なオプション

- `-d, --depth`: 探索する深さ
- `-t, --target`: 最大深さでの目標値
- `--cols`: 列数
- `--cuda`: CuPy/CUDA による popcount を有効化
- `--output`: 結果出力先ファイル
- `--resume`: checkpoint から再開

```powershell
python HLSearch.py --depth 4 --cols 500 --output shift_path.txt
```

## テスト

```powershell
pytest -q

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
