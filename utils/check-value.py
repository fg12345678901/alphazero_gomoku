# 验证 getGameEnded() 与 get_winner() 的符号逻辑
from gomoku.game import GomokuGame
from gomoku.display import display
g = GomokuGame()
b = g.getInitBoard()
# 黑方在 (7,7) 连下 5 子
for i in range(5):
    b.do_move(b.coord_to_move(7,7+i))  # 黑
    display(b)
    if i < 4:
        b.do_move(b.coord_to_move(0,i))  # 白随意落
        display(b)
assert b.get_winner() == 1                    # 黑赢
print(b.get_winner())
assert g.getGameEnded(b, 1) == 1              # 站在黑视角→+1
assert g.getGameEnded(b, -1) == -1            # 站在白视角→-1
print("✔ 符号一致，无颠倒")
