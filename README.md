# 赤い箱をつかんで右へ運ぶ — Genesis / Docker / JupyterLab

既存の公開実演データから小さなニューラルネットを学習し、その推論で
Franka Pandaを動かす教材です。CPUで学習・シミュレーション・動画出力を行います。
GPU、CUDA設定、APIキーは不要です。

## 出典・作成経緯

本課題は公開実演データを利用し、Genesis上で模倣学習と評価を行うものです。外部素材の出典は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に記載しています。学習データ、生成したモデル・動画はリポジトリに含めず、実行時に取得・生成します。

## Windowsで始める

1. Docker Desktopをインストールし、WSL 2バックエンド・Linuxコンテナで起動します。
2. ZIPを展開します。`Dockerfile` と `compose.yaml` のあるフォルダでPowerShellを開きます。
3. 以下を実行します。

```powershell
docker compose up --build
```

初回はPythonパッケージおよび公開データの取得のためネット接続が必要です。データはDockerイメージに含まず、Jupyterノートブックの実行時に取得します。
Genesisの依存関係が多く、ダウンロード・ビルドに時間とディスク容量を使います。
まずDockerにメモリ8 GB程度、空きディスク15 GB程度を確保するのが目安です
（最低動作要件を実測した値ではありません）。

4. ログに出る `http://127.0.0.1:8888/lab?token=...` をブラウザで開きます。
   ホスト名がコンテナIDの場合は、ホスト部分を `127.0.0.1` に置き換えます。
   トークン認証は有効のままです。ポートは自分のPCのみに公開します。
5. `notebooks/01_train_and_run.ipynb` を開き、上から Shift + Enter で実行します。
6. 終了はPowerShellで Ctrl + C。その後、必要に応じて `docker compose down`。
   学習済みモデルや動画は、展開したフォルダの `outputs/` に残ります。

2回目以降は `docker compose up` で起動できます。
ノートブックの標準設定は再学習します。最初は必ず再学習します。5件テスト用のモデルを作成した後は `RETRAIN=False` にできます。

## 何をAIに学ばせるのか

入力は、箱の**初期XY座標・既知の重さ・時刻**。出力は手の目標XYZ位置と把持力です。
PyTorchの小さなMLPを教師あり学習します。学習済み重みを保存し、推論時に読み込みます。
目標位置を関節角に変える逆運動学と、関節を動かすPD制御はGenesisの機能です。

このモデルは初期条件から動作軌道を生成する入門用ベースラインです。
赤色は表示上の指定であり、色を認識して選別しているわけではありません。
重さは本来データ上では隠れた値ですが、この教材ではシミュレーターの既知情報として入力します。
画像認識、言語理解、触覚による滑り検知、途中で動いた箱への追従、失敗からの復帰はありません。

## 既存データと環境

- 出典：<https://huggingface.co/datasets/jalal06/tactile-genesis-heavy-mg-v2>
- 固定revision：`9eb6af694a4a519a8c47f50076af3dcf20ab3e4d`
- 公開済みの実演24件を初回実行時に固定revisionから取得し、数値配列だけの軽量版を作成します。取得にはインターネット接続が必要です。新しく生成した正解動作ではありません。
- 実演単位で学習16件・検証3件・テスト5件。フレーム単位のランダム分割はしません。
- モデル選択は検証損失を使用。テスト実演は誤差・物理動作の評価に使います。
- 元の観測は各ステップの後に記録されるため、`state[t]` と次の `action[t+1]` が対応します。
- 元は灰色の箱。本教材は赤色へ変更し、青い台への配置に簡略化。
  元環境のコードを取得できず、同一物理設定を完全再現したものではありません。
- 床、Panda、4 cm立方体、13 cm角の配置台を再構成。障害物は省略しました。
- 「右」はカメラ基準。カメラが +Y 側にあるため、世界座標 −X 方向が画面右になります。

出典・ライセンス・加工内容は `THIRD_PARTY_NOTICES.md`、取得した各ファイルのチェックサムは
`data/compact/manifest.json` に記録します。元の画像・触覚配列はZIPやDockerイメージには含めません。初回取得した元データは `data/raw/` に保存されます。
必要ならノートブック末尾から同じrevisionの元データを再取得できます。

## 推論と正解再生の違い

```powershell
# JupyterLab起動中、別のPowerShellから学習済みモデルで動かす
docker compose exec lab python -m robot_lab.simulation --mode policy --episode ep0144 --output outputs/policy_ep0144

# 公開データの正解指令を再生する診断モード（推論ではない）
docker compose exec lab python -m robot_lab.simulation --mode replay --episode ep0000 --output outputs/replay_ep0000

# データ分割、時刻対応、保存復元、成功判定の確認
docker compose exec lab python -m unittest discover -s tests -v
```

`policy` はモデル出力をモーターに渡し、`replay` は正解指令を渡します。
失敗したら正解動作に切り替えるフォールバックはありません。
箱の位置を直接書き換えたり、手と箱を固定する拘束を加えたりせず、指の接触力で運びます。
`set_dofs_position` はロボットの初期姿勢設定にだけ使用します。

## 結果の見方

各 `outputs/policy_実演ID/` に以下を保存します。

| ファイル | 内容 |
|---|---|
| `rollout.mp4` | 30 fpsの推論映像。約3.6秒 |
| `initial.png`, `final.png` | 初期・終了画面 |
| `result.json` | 持ち上げ・配置・解放・静止の判定 |
| `trace.csv`, `trace.json` | 毎ステップの箱位置と指令 |
| `commands.npy` | モデルが推論した157ステップの指令 |

成功条件は、高さ10 cm超への持ち上げ、台中心から4.5 cm以内、台上の高さ、
最後の0.5秒で5 mm未満の移動、最後の把持指令が開いた状態、のすべてです。
指令位置の予測誤差と物理タスク成功率は別に確認します。
ソフトウェアのバージョン・CPU・接触計算の差で結果が変わることがあります。

## 1〜3分の提出動画にする

ノートブックを上から実行し、再学習とテスト5件の実行を終えると、5件のロボット動作映像のみを0.25倍速でつないだ約72秒の `outputs/submission_80s.mp4` が生成されます。字幕、説明画面、音声はありません。再学習後は必ず5件とも撮影し直してください。説明は別添レポートを参照してください。

## 困ったとき

- **初回だけ長く止まって見える**：GenesisのCPUカーネルをコンパイルしています。
  数分以上かかる場合があります。`[*]` とログを確認してください。
- **EGL / OpenGLエラー**：同梱のDocker環境を使い、Composeの3つの描画環境変数を確認します。
  `--no-video` で物理演算だけ実行して描画の問題を切り分けられます。
- **メモリ不足**：Dockerのメモリ割り当てを増やし、ほかの重いアプリを閉じます。
- **8888番ポートが使用中**：Compose左側を `127.0.0.1:8889:8888` に変更し、ブラウザは8889番にアクセスします。
- **モデルがうまく運べない**：まずJSONの各条件を確認。学習側実演の `replay` でも失敗するなら
  環境差を疑います。失敗を成功と表示することはありません。
- **再現性を保つ**：データ分割・乱数seed・主要パッケージを固定しています。
  Dockerイメージのdigestと全推移依存までは固定していないため、完全なビット一致は保証しません。

## 参考資料

- Genesis 制御：<https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html>
- Docker Desktop WSL：<https://docs.docker.com/desktop/features/wsl/>
- JupyterLab：<https://jupyterlab.readthedocs.io/en/stable/>

## キャッシュ権限の修正について

この修正版はNumba、Genesis、Quadrantsのキャッシュをコンテナ内の `/tmp/genesis-lab-cache` に設定します。
従来の `/home/learner/.cache` ボリュームの所有者に依存しません。コンテナを作り直すとコンパイルキャッシュは
消えるため、次の実行で再コンパイルが必要になる場合があります。学習データと `outputs/` は作業フォルダに保存されます。

旧版からの切り替えは、旧版のフォルダで `docker compose down`（`-v` は付けない）を実行し、
このZIPを別フォルダに展開して、そのフォルダで `docker compose up --build` を実行してください。
ユーザーが旧フォルダで再学習したモデルを使う場合は、必要に応じて旧 `outputs/model/` を新しいフォルダへコピーしてください。

修正後のDocker Desktop上での起動は未検証です。問題が続く場合は、エラーログ末尾と、どのノートブックを実行したかを確認してください。
