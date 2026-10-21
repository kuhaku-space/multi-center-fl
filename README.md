# mcfl

NIST Special Database 19 の手書き文字データを使った、マルチセンター連合学習 (Multi-Center Federated Learning) のシミュレーション実装です。

複数のグローバルモデル（センター）を用意し、各クライアントがどのセンターに属するかを EM アルゴリズムで推定しながら学習します。手法は論文 [Multi-Center Federated Learning (arXiv:2005.01026)](https://arxiv.org/abs/2005.01026) に基づいています。比較用として、MNIST 上の標準的な FedAvg 実装も含みます。

## 手法の概要

各ラウンドで以下を繰り返します。

1. **クライアントのサンプリング**: 全クライアントのうち `frac` の割合をランダムに選ぶ。
2. **E-step**（[client.py](src/mcfl/client.py)）: 各クライアントが学習データの先頭バッチで全センターの損失を計算し、`softmax(-loss / τ)` でセンターへのソフト割り当て確率を求める。確率が最大のセンターのモデルをコピーしてローカル学習する。
3. **M-step**（[server.py](src/mcfl/server.py)）: 各センターのモデルを、そのセンターへの割り当て確率で重み付けしたクライアント重みの平均で更新する。
4. **評価**: 各クライアントのテストデータで、損失が最小のセンターのモデルを選んで精度を測る。

> 公式実装 ([caifederated/multi-center-fed-learning](https://github.com/caifederated/multi-center-fed-learning)) はクライアント重みに対する K-Means を使っていますが、本実装は論文の記述どおり損失ベースの EM アルゴリズムを採用しています。

## 必要環境

- Python 3.13 以上
- [uv](https://docs.astral.sh/uv/)（[mise](https://mise.jdx.dev/) を使う場合は `mise install` で Python と uv をまとめて導入できます）
- CUDA 対応 GPU（任意。なければ CPU で動作します）

## セットアップ

```bash
# 依存パッケージのインストール
uv sync

# NIST SD19 (by_class.zip) のダウンロードと展開
uv run scripts/setup_data.py
```

データは `./data/nist/extracted/by_class` に展開されます。

## 使い方

### マルチセンター連合学習の実行

```bash
uv run mcfl-train
```

主なオプション:

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--num-centers` | `2` | センター（グローバルモデル）の数 |
| `--rounds` | `50` | 通信ラウンド数 |
| `--local-epochs` | `5` | クライアントごとのローカルエポック数 |
| `--batch-size` | `32` | バッチサイズ |
| `--lr` | `0.05` | 学習率 (SGD) |
| `--frac` | `0.1` | 各ラウンドで参加するクライアントの割合 |
| `--seed` | `42` | 乱数シード |
| `--max-writers` | `100` | 使用する書き手（クライアント）の最大数 |
| `--max-imgs-per-writer` | `100` | 書き手 1 人あたりの最大画像数 |
| `--noniid-type` | `writer` | Non-IID 分割方法。`writer`: 書き手ごとの自然な分割、`dirichlet`: Dirichlet 分布でクラス分布を偏らせる |
| `--alpha` | `0.5` | Dirichlet 分布の集中度（小さいほど偏りが強い） |
| `--data-root` | `./data/nist/extracted` | データのルートディレクトリ |
| `--device` | `cuda`（利用可能な場合） | 使用デバイス |
| `--log-base-dir` | `./logs` | ログの出力先 |

全オプションは `uv run mcfl-train --help` で確認できます。

実行例:

```bash
uv run mcfl-train --num-centers 3 --rounds 100 --noniid-type dirichlet --alpha 0.1
```

各クライアントは 1 人の書き手に対応し、データは 80% を学習用、20% をテスト用に分割します。ラベルは数字・英大文字・英小文字の 62 クラスです。

### 結果の可視化

```bash
uv run streamlit run app.py
```

ブラウザでダッシュボードが開き、実験ごとに以下を確認できます。

- 精度・損失・ラウンドごとの実行時間の推移
- 各センターに所属するクライアント数の推移
- クライアントのセンター割り当てのヒートマップ

### 未完了ログの削除

```bash
uv run scripts/cleanup_logs.py
```

`./logs` 内の実験のうち、設定したラウンド数まで到達していないものを一覧表示し、確認のうえ削除します。

### FedAvg（ベースライン）

```bash
uv run baselines/fedavg.py --model cnn          # Non-IID
uv run baselines/fedavg.py --model cnn --iid    # IID
```

MNIST 上で FedAvg ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)) を実行します。MNIST は初回実行時に `./data/` へ自動でダウンロードされ、精度の推移は `fedavg_result.png` に保存されます。

## ログ出力

実験ごとに `./logs/<YYYYmmdd_HHMMSS>_K<num_centers>_s<seed>/` が作られ（同名がある場合は末尾に `_1`, `_2`, ... が付きます）、以下のファイルが保存されます。

| ファイル | 内容 |
| --- | --- |
| `config.json` | 実験の設定 |
| `train.log` | コンソールに出力されたログ（例外のトレースバックを含む） |
| `log.csv` | ラウンドごとの学習損失・検証損失・精度・実行時間・各センターの所属クライアント数 |
| `client_allocation.csv` | ラウンドごとの各クライアントの所属センター |

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| [src/mcfl/train.py](src/mcfl/train.py) | マルチセンター連合学習のエントリーポイント（`mcfl-train`） |
| [src/mcfl/config.py](src/mcfl/config.py) | 設定とコマンドライン引数 |
| [src/mcfl/dataset.py](src/mcfl/dataset.py) | NIST データの読み込みとクライアントへの分割 |
| [src/mcfl/model.py](src/mcfl/model.py) | CNN モデル |
| [src/mcfl/client.py](src/mcfl/client.py) | クライアント（E-step とローカル学習） |
| [src/mcfl/server.py](src/mcfl/server.py) | サーバー（M-step と評価） |
| [src/mcfl/experiment_logger.py](src/mcfl/experiment_logger.py) | 実験ログの保存 |
| [app.py](app.py) | Streamlit ダッシュボード |
| [scripts/setup_data.py](scripts/setup_data.py) | NIST データのダウンロード |
| [scripts/cleanup_logs.py](scripts/cleanup_logs.py) | 未完了ログの削除 |
| [baselines/fedavg.py](baselines/fedavg.py) | MNIST での FedAvg ベースライン |

いずれのコマンドもプロジェクトルートで実行してください（`./data` と `./logs` はカレントディレクトリからの相対パスです）。
