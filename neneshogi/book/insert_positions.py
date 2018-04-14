"""
棋譜から局面を取り出してdbに挿入

棋譜:
startpos moves 7g7f ...
"""

import argparse
import os
import sqlite3
import subprocess
from collections import defaultdict

from typing import List, Dict


class TreeNode:
    children: Dict[str, "TreeNode"]
    count: int

    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.count = 0

    def add(self, moves: List[str]):
        self.count += 1
        if len(moves) == 0:
            return
        move = moves.pop(0)
        child = self.children[move]
        child.add(moves)

    def iterate(self, parent_moves: str = ""):
        yield (parent_moves, self.count)  # ルートノードも考察しないといけないので、parent_moves==""も出力
        if parent_moves:
            # not root node
            parent_moves += " "
        for move, node in self.children.items():
            for item in node.iterate(parent_moves + move):
                yield item


def generate_tree(kifu: str, maxlength: int):
    root = TreeNode()
    with open(kifu, "r") as f:
        for line in f:
            moves = line.rstrip().split(" ")[2:maxlength + 2]
            root.add(moves)
    return root


def insert_to_db(db_path: str, tree: TreeNode, mincount: int):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    items = []
    for moves, count in tree.iterate():
        if count >= mincount:
            items.append((moves, count))
            if len(items) >= 10000:
                cur.executemany("INSERT INTO book(moves, count) VALUES(?, ?)", items)
                items = []
                con.commit()
    if len(items) > 0:
        cur.executemany("INSERT INTO book(moves, count) VALUES(?, ?)", items)
        items = []
    con.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    parser.add_argument("kifu")
    parser.add_argument("--mincount", type=int, default=2)
    parser.add_argument("--maxlength", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print("making initial db")
        subprocess.check_call(["python", "-m", "neneshogi.book.init_db", args.db])

    print("generating tree")
    tree = generate_tree(args.kifu, args.maxlength)
    print("writing to db")
    insert_to_db(args.db, tree, args.mincount)


if __name__ == '__main__':
    main()
