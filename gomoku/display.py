# gomoku/display.py
from .board import Board

def display(board: Board):
    print(board)
    print('当前执子:', '黑' if board.current_player == 1 else '白')
