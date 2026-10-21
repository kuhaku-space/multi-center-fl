# mcfl

NIST Special Database 19 の手書き文字データを使った、マルチセンター連合学習 (Multi-Center Federated Learning) のシミュレーション実装です。

複数のグローバルモデル（センター）を用意し、各クライアントがどのセンターに属するかを EM アルゴリズムで推定しながら学習します。手法は論文 [Multi-Center Federated Learning (arXiv:2005.01026)](https://arxiv.org/abs/2005.01026) に基づいています。比較用として、MNIST 上の標準的な FedAvg 実装も含みます。

## 手法の概要

学習の前に、センターの初期値を作ります（[init_centers.py](src/mcfl/init_centers.py)、`--pretrain-rounds 0` で無効化してランダム初期化）。

1. 共通モデルを FedAvg で `pretrain_rounds` ラウンド事前学習する。
2. 全クライアントが事前学習モデルからローカル学習し、学習後の重み（事前学習モデルからの差分）を K-Means でクラスタリングする。論文と同様に、初期値を変えて20回クラスタリングし、クラスタ内距離が最小の結果を使う。
3. 各クラスタに属するクライアントの重みの平均を、各センターの初期値とする。

その後、各ラウンドで以下を繰り返します。

1. **クライアントのサンプリング**: 全クライアントのうち `frac` の割合をランダムに選ぶ（`--sampling center` では所属センターごとに `frac` ずつ選ぶ）。論文は全クライアントが毎ラウンド参加する設定（`--frac 1.0`）。
2. **ローカル学習と E-step**: `--assignment` で割り当て方法を選ぶ。
   - `distance`（デフォルト、[server.py](src/mcfl/server.py)）: 各クライアントが所属センター（最初は K-Means の初期クラスタ）のモデルからローカル学習し、学習後の重みに重み空間で最も近いセンターに割り当て直す。論文の E-step (Eq. 8) に対応する。
   - `loss`（[client.py](src/mcfl/client.py)）: 各クライアントが学習データ全体で全センターの平均損失を計算し、`softmax(-loss / τ)` が最大（= 損失が最小）のセンターからローカル学習し、そのセンターに割り当てる。論文が比較手法として挙げている HypoCluster / IFCA と同様の方法で、学習の進んだセンターが全クライアントを取り込みやすい。
   - `--prox-mu` が正なら、ローカル学習の損失に学習開始時のセンターからの距離の近接項を加える（論文の Eq. 7）。
3. **M-step**（[server.py](src/mcfl/server.py)）: 各センターのモデルを、そのセンターに割り当てられたクライアントの重みの平均で更新する。割り当てられたクライアントがいないセンターは前のモデルを維持する。
4. **評価**: 各クライアントのテストデータを、所属センターのモデルで評価する（`loss` モードでは、E-step と同じく学習データ全体で損失が最小のセンターを選ぶ）。`--finetune-epochs` が正なら、センターのモデルのコピーをクライアントの学習データでローカル学習してから評価する。

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
| `--method` | `mcfl` | 学習方法（`mcfl`: マルチセンター連合学習、`local`: 各クライアントが自分のデータだけで学習する比較用の Local only、`cfl`: 比較用の CFL） |
| `--num-centers` | `2` | センター（グローバルモデル）の数 |
| `--rounds` | `50` | 通信ラウンド数 |
| `--local-epochs` | `5` | クライアントごとのローカルエポック数 |
| `--batch-size` | `32` | バッチサイズ |
| `--lr` | `0.05` | 学習率 (SGD) |
| `--frac` | `0.1` | 各ラウンドで参加するクライアントの割合 |
| `--sampling` | `global` | サンプリング方法（`global`: 全体から `frac` を抽出、`center`: 所属センターごとに `frac` ずつ抽出。`--assignment distance` のみ） |
| `--seed` | `42` | 乱数シード |
| `--max-writers` | `100` | 使用する書き手（クライアント）の最大数 |
| `--max-imgs-per-writer` | `100` | 書き手 1 人あたりの最大画像数 |
| `--noniid-type` | `writer` | Non-IID 分割方法。`writer`: 書き手ごとの自然な分割、`dirichlet`: Dirichlet 分布でクラス分布を偏らせる |
| `--cluster-shift` | `none` | 正解のあるクラスタを人工的に作る変換。クライアントを `--num-groups` 個のグループにランダムに分け、グループ 0 以外に変換を加える（`rotation`: グループ g の画像を 90g 度回転、`label`: グループごとに固定したランダムな置換でラベルを入れ替える） |
| `--num-groups` | `4` | `--cluster-shift` のグループ数（`rotation` は 4 まで） |
| `--assignment` | `distance` | E-step の割り当て方法（`distance`: 学習後の重みに最も近いセンター、`loss`: 損失が最小のセンター） |
| `--prox-mu` | `0` | ローカル学習で損失に加える近接項 (μ/2)‖w − w_center‖² の係数（論文の Eq. 7）。`0` で無効 |
| `--pretrain-rounds` | `5` | センター初期化前の FedAvg 事前学習ラウンド数（`0` でランダム初期化） |
| `--cfl-eps1` | `0.4` | CFL: クラスタの更新の平均のノルムがこれより小さいとき分割する |
| `--cfl-eps2` | `1.6` | CFL: クラスタ内の更新のノルムの最大がこれより大きいとき分割する |
| `--cfl-warmup-rounds` | `20` | CFL: 最初のこのラウンド数の間は分割しない |
| `--finetune-epochs` | `0` | 評価前に、所属センターのモデルのコピーを各クライアントの学習データでこのエポック数だけローカル学習する（個別化の評価。センターは変更しない）。`0` で無効 |
| `--alpha` | `0.5` | Dirichlet 分布の集中度（小さいほど偏りが強い） |
| `--data-root` | `./data/nist/extracted` | データのルートディレクトリ |
| `--device` | `cuda`（利用可能な場合） | 使用デバイス |
| `--log-base-dir` | `./logs` | ログの出力先 |
| `--save-models` | 無効 | 学習完了時に各センターのモデルを `checkpoints/` に保存する |

全オプションは `uv run mcfl-train --help` で確認できます。

実行例:

```bash
uv run mcfl-train --num-centers 3 --rounds 100 --noniid-type dirichlet --alpha 0.1
```

各クライアントは 1 人の書き手に対応し、データは 80% を学習用、20% をテスト用に分割します。ラベルは数字・英大文字・英小文字の 62 クラスです。

### 比較手法（NIST 上のベースライン）

マルチセンター連合学習と同じデータ・CNN・クライアント分割で、以下のベースラインを実行できます。

| 手法 | コマンド |
| --- | --- |
| FedAvg ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)) | `uv run mcfl-train --num-centers 1 --pretrain-rounds 0` |
| FedProx ([Li et al., 2020](https://arxiv.org/abs/1812.06127)) | `uv run mcfl-train --num-centers 1 --pretrain-rounds 0 --prox-mu 0.01` |
| IFCA ([Ghosh et al., 2020](https://arxiv.org/abs/2006.04088)) | `uv run mcfl-train --assignment loss --pretrain-rounds 0` |
| CFL ([Sattler et al., 2020](https://arxiv.org/abs/1910.01991)) | `uv run mcfl-train --method cfl --frac 1.0 --num-centers 10` |
| Local only | `uv run mcfl-train --method local` |
| FedAvg + fine-tuning | `uv run mcfl-train --num-centers 1 --pretrain-rounds 0 --finetune-epochs 5` |

- センターが 1 つなら、M-step は参加クライアントの重みの平均になり、FedAvg と一致する。ただし平均はサンプル数で重み付けしない単純平均（マルチセンター連合学習の M-step と同じ）。
- マルチセンター連合学習は、`--rounds` とは別に `--pretrain-rounds` ラウンドの FedAvg 事前学習と、全クライアントによる 1 回のローカル学習（K-Means 用）を行う。通信量を揃えて比較する場合は、FedAvg の `--rounds` をその分増やす。
- IFCA は IFCA の model averaging 版に相当する（全センターのモデルを配信し、各クライアントが学習データ全体の損失が最小のセンターを選んでローカル学習し、センターごとに平均する。センターはランダム初期化）。毎ラウンド K 個のモデルを配信するため、下り通信量は FedAvg や距離ベースの E-step の K 倍になる。`--pretrain-rounds` を正にすれば、本実装と同じ K-Means 初期化を使った IFCA になる。
- CFL は 1 つのクラスタ（ランダム初期化）から始めて各クラスタで FedAvg を行い、クラスタの更新の平均のノルムが `--cfl-eps1` より小さく、更新のノルムの最大が `--cfl-eps2` より大きいクラスタを、更新のコサイン類似度による階層クラスタリング（complete linkage）で 2 つに分割する（公式実装と同じ判定）。クラスタ数は事前に決めず、`--num-centers` は上限として使う。分割に全メンバーの更新が必要なため `--frac 1.0` が必須。ノルムの大きさはモデルや学習率に依存するので、`train.log` の各クラスタの `Mean norm` / `Max norm` と `client_round.csv` の `Update Norm` を見て閾値を調整する。
- Local only は通信せず、`--frac` によらず毎ラウンド全クライアントが自分のモデルを `--local-epochs` エポック学習し、自分のテストデータで評価する。`--num-centers` と `--pretrain-rounds` は無視される。
- `--finetune-epochs` はマルチセンター連合学習とも組み合わせられる（各クライアントが所属センターのモデルから fine-tuning する）。

### 結果の可視化

```bash
uv run streamlit run app.py
```

ブラウザでダッシュボードが開き、実験ごとに以下を確認できます。

- 実験の終了状態・設定・実行環境
- 精度・損失・ラウンドごとの実行時間（学習・評価）の推移
- センターごとの精度の推移
- 各センターに所属するクライアント数の推移
- 割り当て確率のエントロピー（センターをどれだけはっきり区別できているか）の推移（`loss` モード）
- 学習後に所属センターが変わったクライアントの割合の推移（`distance` モード）
- クライアントごとの精度の分布
- クライアントのセンター割り当てのヒートマップ

### 未完了ログの削除

```bash
uv run scripts/cleanup_logs.py
```

`./logs` 内の実験のうち、完了していないもの（`summary.json` の状態が `completed` でないもの。`summary.json` がなければ設定したラウンド数まで到達していないもの）を一覧表示し、確認のうえ削除します。

### FedAvg（ベースライン）

```bash
uv run baselines/fedavg.py --model cnn          # Non-IID
uv run baselines/fedavg.py --model cnn --iid    # IID
```

MNIST 上で FedAvg ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)) を実行します。MNIST は初回実行時に `./data/` へ自動でダウンロードされ、精度の推移は `fedavg_result.png` に保存されます。

## ログ出力

実験ごとに `./logs/<YYYYmmdd_HHMMSS>_K<num_centers>_s<seed>/` が作られ（`--method` が `mcfl` 以外では末尾に `_local` などの手法名が付き、同名がある場合は末尾に `_1`, `_2`, ... が付きます）、以下のファイルが保存されます。

| ファイル | 内容 |
| --- | --- |
| `config.json` | 実験の設定 |
| `meta.json` | 実行環境（コマンド、ホスト名、Python・PyTorch・CUDA のバージョン、GPU、git のコミットと未コミット変更の有無） |
| `git_diff.patch` | 未コミットの変更がある場合のみ、`git diff HEAD` の内容 |
| `summary.json` | 終了状態（`completed` / `failed` / `interrupted`）、実行時間、最終ラウンドと最高精度のラウンドの指標、失敗時のトレースバック |
| `train.log` | コンソールに出力されたログ（例外のトレースバックを含む） |
| `log.csv` | ラウンドごとの学習損失・検証損失・精度（`Accuracy`: サンプル数で重み付けした micro、`Macro Accuracy`: クライアントごとの精度の平均）・実行時間（合計、学習、評価）・各センターの所属クライアント数・センターごとの精度と検証損失 |
| `clients.csv` | 各クライアントの学習データ数・テストデータ数・K-Means による初期クラスタ（`Init Center`、事前学習時のみ）・正解のグループ（`Group`、`--cluster-shift` 指定時のみ） |
| `client_round.csv` | 縦持ち形式。ラウンドごとに参加したクライアントの学習を始めたセンター (`Start Center`)、割り当てたセンター (`Center`)、割り当ての根拠（`loss`: 各センターへの割り当て確率 `P_k` とエントロピー、`distance`: 各センターとの距離 `D_k`、`cfl`: 更新のノルム `Update Norm`）、学習損失 |
| `client_eval.csv` | 縦持ち形式。ラウンドごと・クライアントごとの評価結果（評価に使ったセンター、精度、検証損失、テストデータ数） |
| `client_allocation.csv` | ラウンドごとの各クライアントの所属センター（行: ラウンド、列: クライアント） |
| `checkpoints/center_<k>.pt` | `--save-models` 指定時のみ、学習完了時の各センターのモデルの `state_dict` |

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| [src/mcfl/train.py](src/mcfl/train.py) | マルチセンター連合学習と比較手法のエントリーポイント（`mcfl-train`） |
| [src/mcfl/config.py](src/mcfl/config.py) | 設定とコマンドライン引数 |
| [src/mcfl/dataset.py](src/mcfl/dataset.py) | NIST データの読み込みとクライアントへの分割 |
| [src/mcfl/model.py](src/mcfl/model.py) | CNN モデル |
| [src/mcfl/client.py](src/mcfl/client.py) | クライアント（E-step とローカル学習） |
| [src/mcfl/server.py](src/mcfl/server.py) | サーバー（M-step と評価） |
| [src/mcfl/cfl.py](src/mcfl/cfl.py) | CFL のサーバー（クラスタの分割） |
| [src/mcfl/init_centers.py](src/mcfl/init_centers.py) | FedAvg 事前学習と K-Means によるセンターの初期化 |
| [src/mcfl/experiment_logger.py](src/mcfl/experiment_logger.py) | 実験ログの保存 |
| [app.py](app.py) | Streamlit ダッシュボード |
| [scripts/setup_data.py](scripts/setup_data.py) | NIST データのダウンロード |
| [scripts/cleanup_logs.py](scripts/cleanup_logs.py) | 未完了ログの削除 |
| [baselines/fedavg.py](baselines/fedavg.py) | MNIST での FedAvg ベースライン（NIST 上の比較は上記の「比較手法」を参照） |

いずれのコマンドもプロジェクトルートで実行してください（`./data` と `./logs` はカレントディレクトリからの相対パスです）。
