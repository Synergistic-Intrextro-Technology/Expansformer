import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from solver import solve

print(solve(sys.stdin.read()))
