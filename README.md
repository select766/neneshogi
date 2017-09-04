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
