"""
思考エンジン(抽象クラス)
"""


from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, Tuple, Optional

from .usi_info_writer import UsiInfoWriter


class Engine(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def author(self) -> str:
        pass

    @abstractmethod
    def get_options(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def isready(self, options: Dict[str, str]) -> None:
        pass

    def usinewgame(self) -> None:
        pass

    @abstractmethod
    def position(self, command: str) -> None:
        pass

    @abstractmethod
    def go(self, usi_info_writer: UsiInfoWriter, btime: Optional[int]=None, wtime: Optional[int]=None,
           byoyomi: Optional[int]=None, binc: Optional[int]=None, winc: Optional[int]=None) -> str:
        pass

    def gameover(self, result: str) -> None:
        pass

    def quit(self) -> None:
        pass
