# 外部データ・ソフトウェアの出典

## 公開データ

- 配布元：jalal06 / tactile-genesis-heavy-mg-v2
- データセットカード：<https://huggingface.co/datasets/jalal06/tactile-genesis-heavy-mg-v2>
- 取得するrevision：`9eb6af694a4a519a8c47f50076af3dcf20ab3e4d`
- データセットカードの表示：MIT（2026年10月31日に確認）。権利者・元データに追加条件がないかは配布元で確認すること。
- 加工：24実演を取得し、画像・触覚配列を除いて数値データを保存する。24件を16件の学習、3件の検証、5件のテストに分割する。
- 元データ：Genesis上でPandaが灰色の箱を障害物の間からバスケットに運ぶ実演。本課題：赤い箱を青い台へ運ぶ簡略化した環境。元のデータや学習済みモデルはリポジトリに含めない。
- データセットカードが案内する収集・学習コード：<https://github.com/jalaln06/tectile_genesis>。本課題のMLPはデータから学習する小規模モデルであり、同コードのDiT学習パイプラインを使用したと主張しない。

取得データやその派生物を別途再配布するときは、配布元のライセンスと必要な著作権表示を確認し、適切に添付する。ここに書いたライセンス情報は、個々のデータや別の構成要素に一律に適用されるという意味ではない。

## Genesis とロボットモデル

- Genesis World：<https://github.com/Genesis-Embodied-AI/genesis-world>。Genesis本体のソースコードはApache-2.0と表示されている。
- GenesisのPandaモデルはインストールしたパッケージ側から使用する。モデルやアセット、描画出力の公開条件は、本体コードと別に元の配布物で確認すること。
- 関連する上流モデル：<https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda>。Menagerieの一覧では当該モデルをApache-2.0と表示しているが、本プロジェクトでインストールされる個々のアセットとの対応・追加条件までは確認できていない。
- Franka Pandaはモデルを識別するための名称であり、権利者による推奨を意味しない。

## その他の依存関係

PyTorch、NumPy、JupyterLab等はインストール時に取得する。直接依存のバージョンは `requirements.txt` と `Dockerfile` に記載。各パッケージおよび推移的依存のライセンスはそれぞれ異なるため、イメージや依存物を再配布する場合は各配布元を確認すること。
