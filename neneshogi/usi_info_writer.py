from typing import Optional, Iterable, Callable

from .position import Move


class UsiInfoWriter:
    lines_writer: Callable[[Iterable[str]], None]

    def __init__(self, lines_writer: Callable[[Iterable[str]], None]):
        self.lines_writer = lines_writer

    def write_pv(self, *, pv: Iterable[Move], depth: Optional[int] = None, seldepth: Optional[int] = None,
                 time: Optional[int] = None, nodes: Optional[int] = None, score_cp: Optional[int] = None,
                 score_mate: Optional[int] = None, score_mate_unknown: bool = False,
                 ):
        items = ["info"]

        def add_value_if_not_none(name, value):
            if value is not None:
                items.append(name)
                items.append(str(int(value)))

        add_value_if_not_none("depth", depth)
        add_value_if_not_none("seldepth", seldepth)
        add_value_if_not_none("time", time)
        add_value_if_not_none("nodes", nodes)
        if time is not None and time > 0 and nodes is not None:
            items.append("nps")
            items.append(str(int(nodes * 1000 / time)))

        if score_cp is not None:
            items.extend(["score", "cp", str(int(score_cp))])
        elif score_mate is not None:
            if score_mate_unknown:
                items.extend(["score", "mate", ("+" if score_mate > 0 else "-")])
            else:
                items.extend(["score", "mate", str(int(score_mate))])

        items.append("pv")
        items.extend([move.to_usi_string() for move in pv])

        self.lines_writer([" ".join(items)])

    def write_string(self, message: str):
        self.lines_writer([f"info string {message}"])
