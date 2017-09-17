import sys
from typing import Optional, Iterable

from .position import Move


class UsiInfoWriter:
    def __init__(self):
        pass

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

        if score_cp is not None:
            items.extend(["score", "cp", str(int(score_cp))])
        elif score_mate is not None:
            if score_mate_unknown:
                items.extend(["score", "mate", ("+" if score_mate > 0 else "-")])
            else:
                items.extend(["score", "mate", str(int(score_mate))])

        items.append("pv")
        items.extend([move.to_usi_string() for move in pv])

        info_str = " ".join(items) + "\n"
        sys.stdout.write(info_str)
        sys.stdout.flush()

    def write_string(self, message: str):
        sys.stdout.write(f"info string {message}\n")
        sys.stdout.flush()
