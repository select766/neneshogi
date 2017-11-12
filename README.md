# ねね将棋 (Nene Shogi)
NEural NEtwork Shogi

Deep Learningを用いた評価関数で探索を行う将棋プログラム。第5回将棋電王トーナメント(2017年11月)版。予選にて、42チーム中32位。

# セットアップ
先に環境構築が必要。手数がかかるので、なんでもいいから将棋プログラムで遊びたいという方には向かない。

Python 3.6以上が必要(それ以前では、文法エラーとなる)。

`import neneshogi` を可能にする
```
python setup.py develop
```

C++のモジュール(`neneshogi_cpp.pyd`)をビルドする場合、pybind11の取得が必要。一応ビルド済みのバイナリが入れてあるので実行するだけなら不要。
```
git submodule init
git submodule update
```

この後、Visual Studio 2017で`neneshogi_cpp\neneshogi_cpp.sln`を開いてビルド。

## 第5回将棋電王トーナメント(2017年11月)版パラメータ
将棋所から利用する。

エンジンとして、`player_bat\monte_carlo_softmax_v2_player.bat`を指定。

定跡ファイルは
[https://github.com/yaneurao/YaneuraOu/releases/tag/v4.73_book](https://github.com/yaneurao/YaneuraOu/releases/tag/v4.73_book)
から入手した`standard_book.db`を`book`ディレクトリに入れる。(定跡はなくても動作可能)

エンジン設定
- `model_path`: `data\model\20170917200841\weight\model_iter_3906250`
- `book`: `book/standard_book.db`
- `gpu`: `0`
- `max_nodes`: `1000` (機能しない)
- `qsearch_depth`: `3`
- `batch_size`: `256`
- `queue_size`: `16`
- `cp_scale`: `150`
- `time_divider`: `50`
- `time_inc_divider`: `25` (電王トーナメントでは機能しない)
- `softmax_temperature`: `0.02`
- `no_ponder`: `false`

GPUの性能が低い環境では、`batch_size`、`queue_size`を減らさないとクラッシュまたは持ち時間を守らないことがある。

# テスト
## 合法手生成のテストケース作成

```
python -m neneshogi.generate_position_testcase data\testcase\generate_position_testcase_kifu.txt
```

## テスト実行
```
python -m unittest
```

## モデル学習
36byte/record(32: packed sfen, 2: value, 2: pv move)の棋譜を用意(最新版のやねうら王だと違うので、2016年末ごろのものに対してdefine設定が必要)

```
python -m neneshogi.train_config clone xxxx  # xxxxは設定クローン元のID
python -m neneshogi.train data\model\yyyy  # yyyyは学習ID
```

# 環境構築
Windows 10, NVIDIA GPUがある環境を想定。設定で`gpu=-1`を指定すればGPUなしでも動かせる。ただし非常に重い。

## 実行用
- NVIDIAドライバ
- CUDA 8.0
- cuDNN
- [Visual C++ 2015 Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools)
  - CUDAの実行時コンパイルに必要
  - `C:\Program Files (x86)\Windows Kits\8.1\bin\x86` から `C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64` へ
    - `rc.exe`, `rcdll.dll` をコピー
  - PATHに以下を追加
    - C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin
- Anaconda (Python 3.6)
- pipにて以下のパッケージ
  - chainer (2.0系)
  - cupy
- 将棋所

## 開発用
- PyCharm
- Visual Studio 2017 Community (C++ Project)
- [Git](https://git-for-windows.github.io/)
