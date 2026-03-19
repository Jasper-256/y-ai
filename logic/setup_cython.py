"""Build script for Cython modules (board, game, mcts)."""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        ["board.pyx", "game.pyx", "mcts.pyx"],
        language_level="3",
    ),
)
