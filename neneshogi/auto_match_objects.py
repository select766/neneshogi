from typing import Dict, List
import yaml


def yaml_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


class Rule:
    n_match: int
    max_moves: int
    priority_low: bool
    max_go_time: float

    def __init__(self):
        self.n_match = 1
        self.max_moves = 256
        self.priority_low = False
        self.max_go_time = 60

    @classmethod
    def load(cls, rule_file) -> "Rule":
        item_dict = yaml_load(rule_file)
        inst = cls()
        inst.__dict__.update(item_dict)
        return inst


class EngineConfig:
    path: str
    go: str
    options: Dict[str, str]
    env: Dict[str, str]

    def __init__(self):
        self.path = None
        self.go = "go btime 0 wtime 0 byoyomi 1000"
        self.options = {"USI_Ponder": "false", "USI_Hash": "256"}
        self.env = {}

    @classmethod
    def load(cls, engine_file) -> "Engine":
        item_dict = yaml_load(engine_file)
        inst = cls()
        inst.__dict__.update(item_dict)
        return inst


class MatchResult:
    draw: bool
    winner: int
    gameover_reason: str
    kifu: List[str]

    def __init__(self, draw, winner, gameover_reason, kifu):
        self.draw = draw
        self.winner = winner
        self.gameover_reason = gameover_reason
        self.kifu = kifu


class AutoMatchResult:
    rule: Rule
    engine_config_list: List[EngineConfig]
    match_results: List[MatchResult]
