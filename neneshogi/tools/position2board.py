"""
position startpos moves ...
またはsfenを可視化
"""
import sys
import shogi

def tokens_to_board(tokens):
    board = shogi.Board()
    while len(tokens) > 0:
        token = tokens.pop(0)
        if token == "position":
            continue
        if token == "moves":
            continue
        if token == "startpos":
            board.reset()
            continue
        if token == "sfen":
            # 後続がsfen
            sfen_str = " ".join(tokens[:4])
            del tokens[:4]
            board.set_sfen(sfen_str)
            continue
        if len(token) > 10:
            # sfenと思われる
            sfen_str = " ".join([token] + tokens[:3])
            del tokens[:3]
            board.set_sfen(sfen_str)
            continue
        move_obj = shogi.Move.from_usi(token)
        if move_obj not in board.generate_legal_moves():
            print(f"Warning: {token} is illegal.")
        board.push_usi(token)
    return board


def main():
    for line in sys.stdin:
        tokens = line.rstrip().split(" ")
        board = tokens_to_board(tokens)
        print(board.kif_str())

if __name__ == "__main__":
    main()
