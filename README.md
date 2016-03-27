# DQN_pendulum

Swinging up a pendulm by Deep Q Network

<img src="https://raw.githubusercontent.com/ashitani/DQN_pendulum/master/best.gif">

詳細は[Qiita](http://qiita.com/ashitani/items/bb393e24c20e83e54577)を参照のこと。

## Requirement

- chainer
- Jupyter,matplotlib,numpy
- svgwrite(Jupyterでの確認用なので関連コードを除去すれば不要)
- imageio(アニメ生成用)

## Usage

```
python dqn_pendulum.py
```

model/以下に1000回おきのモデルとハイスコアを出したモデルが生成されます。
収束状況のログはlog.csvに書き込まれます。

```
python make_gif.py model/000000.model
```

などで、アニメーションgifとプロファイルのpngが生成されます。

