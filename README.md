# neneshogi
NEural NEtwork Shogi

# セットアップ
Python 3.6以上が必要(それ以前では、文法エラーとなる)。

`import neneshogi` を可能にする
```
python setup.py develop
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
- 将棋所

## 開発用
- PyCharm
- [Git](https://git-for-windows.github.io/)

