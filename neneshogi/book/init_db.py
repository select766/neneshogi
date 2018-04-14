"""
定跡DBの初期化
"""

import argparse
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    cur.execute(
        "CREATE TABLE book(id INTEGER PRIMARY KEY, moves TEXT UNIQUE NOT NULL, count INTEGER NOT NULL, best_move TEXT, process_id INTEGER)")
    con.commit()


if __name__ == '__main__':
    main()
