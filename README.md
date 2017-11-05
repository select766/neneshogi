# neneshogi
NEural NEtwork Shogi

# セットアップ
Python 3.6以上が必要(それ以前では、文法エラーとなる)。

`import neneshogi` を可能にする
```
python setup.py develop
```

cythonコンパイル(pyxファイルを更新するごとに必要)
```
python setup.py build_ext --inplace
```

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
36byte/record(32: packed sfen, 2: value, 2: pv move)の棋譜を用意(最新版のやねうら王だと違うので、後で改良)

```
python -m neneshogi.train_config clone xxxx  # xxxxは設定クローン元のID
python -m neneshogi.train data\model\yyyy  # yyyyは学習ID
```

# 環境構築
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
  - graphviz
- 将棋所

## 開発用
- PyCharm
- [Git](https://git-for-windows.github.io/)
- [Graphviz](http://www.graphviz.org/Download_windows.php)
  - PATHを手動で通す必要あり。 `C:\Program Files (x86)\Graphviz2.38\bin`
