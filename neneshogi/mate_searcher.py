"""
詰み探索用ルーチン
シンプルに反復深化探索する。
子プロセスとして生成される
"""

from multiprocessing import Process, Value, Queue

from logging import getLogger
from typing import List, Optional

logger = getLogger(__name__)

from .move import Move

from .position import Position


class MateSearcherCommand:
    command: str
    params: object

    def __init__(self, command: str, params: object):
        self.command = command
        self.params = params

    @classmethod
    def go(cls, position: Position) -> "MateSearcherCommand":
        return cls("go", position)

    @classmethod
    def quit(cls) -> "MateSearcherCommand":
        return cls("quit", None)


class MateSearcherResponse:
    command: str
    params: object

    def __init__(self, command: str, params: object):
        self.command = command
        self.params = params

    @classmethod
    def mate_result(cls, win_move: Optional[Move], lose_moves: List[Move]) -> "MateSearcherResponse":
        return cls("mate_result", {"win_move": win_move, "lose_moves": lose_moves})


class MateSearcherMain:
    stop_signal: Value
    command_queue: Queue
    response_queue: Queue
    pos: Position
    win_move: Optional[Move]
    lose_moves: List[Move]

    def __init__(self, stop_signal: Value, command_queue: Queue, response_queue: Queue):
        self.stop_signal = stop_signal
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.pos = None
        self.win_move = None
        self.lose_moves = []

    def run(self):
        while True:
            cmd = self.command_queue.get()  # type: MateSearcherCommand
            if cmd.command == "go":
                self.go(cmd.params)
            elif cmd.command == "quit":
                logger.info("quit command")
                break

    def go(self, params):
        self.pos = params
        depth = 1
        self.last_win_move = None
        self.last_lose_moves = []
        while not self.stop_signal.value:
            self.win_move = None
            self.lose_moves = []
            logger.info(f"searching depth {depth}")
            try:
                val = self.search(True, depth)
                logger.info(f"mate value={val}")
                self.last_win_move = self.win_move
                self.last_lose_moves = self.lose_moves
            except RuntimeError:
                logger.info(f"search interrupted")
                break
            depth += 1
        self.response_queue.put(MateSearcherResponse.mate_result(self.last_win_move, self.last_lose_moves))

    def search(self, is_root: bool, remain_depth: int) -> int:
        """
        現局面が詰みなら、負の値を返す。

        :param remain_depth:
        :return:
        """
        if self.stop_signal.value:
            raise RuntimeError
        moves = self.pos.generate_move_list()
        if len(moves) == 0:
            return -1
        if remain_depth <= 0:
            return 0  # 詰みかどうか不明
        max_child_val = -10
        for move in moves:
            undo_info = self.pos.do_move(move)
            child_val = -self.search(False, remain_depth - 1)
            self.pos.undo_move(undo_info)
            if is_root:
                if child_val > 0:
                    self.win_move = move
                elif child_val < 0:
                    self.lose_moves.append(move)
            if child_val > max_child_val:
                max_child_val = child_val
                if child_val > 0:
                    # これを指せば相手を詰ませられるので、ほかの手は見なくていい
                    break
        return max_child_val


def mate_searcher_main(stop_signal: Value, command_queue: Queue, response_queue: Queue):
    MateSearcherMain(stop_signal, command_queue, response_queue).run()


class MateSearcher:
    stop_signal: Value
    command_queue: Queue
    response_queue: Queue
    process: Process

    def __init__(self):
        self.stop_signal = Value("i", 0)
        self.stop_signal.value = 0
        self.command_queue = Queue()
        self.response_queue = Queue()
        self.process = Process(target=mate_searcher_main,
                               args=(self.stop_signal, self.command_queue, self.response_queue))

    def run(self):
        self.process.start()

    def quit(self):
        self.command_queue.put(MateSearcherCommand.quit())
        self.process.join()
