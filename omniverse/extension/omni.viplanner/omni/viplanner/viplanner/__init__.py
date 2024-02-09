# from .vip_anymal import VIPlanner
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

from .viplanner_algo import VIPlannerAlgo

__all__ = ["DATA_DIR", "VIPlannerAlgo"]
