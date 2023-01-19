import copy
import random
import time
import numpy as np

infinity = 2147483647
MAX_DEPTH = 2
UPPER_DEPTH = 9
STAGE = 0
MAX_TIMEOUT = 4.8
CUT_BY_TIME = False
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
direction_row = [0, 1, 1, 1, 0, -1, -1, -1]
direction_col = [-1, -1, 0, 1, 1, 1, 0, -1]
directions = [(1, 0), (-1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (1, 1), (-1, 1)]
corner_position = [(0, 0), (0, 7), (7, 0), (7, 7)]
c_position = [(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)]
x_position = [(1, 1), (1, 6), (6, 1), (6, 6)]

weight = [[-8000, 260, -50, -30, -30, -50, 260, -8000],
          [260, 100, -10, -5, -5, -10, 100, 260],
          [-50, -10, -5, 3, 3, -5, -10, -50],
          [-30, -5, 3, 1, 1, 3, -5, -30],
          [-30, -5, 3, 1, 1, 3, -5, -30],
          [-50, -10, -5, 3, 3, -5, -10, -50],
          [260, 100, -10, -5, -5, -10, 100, 260],
          [-8000, 260, -50, -30, -30, -50, 260, -8000]]


# weight = [[-500, 150, -200, -30, -30, -200, 150, -500],
#           [150, 100, -10, -5, -5, -10, 100, 150],
#           [-200, -10, -5, 3, 3, -5, -10, -200],
#           [-30, -5, 3, 1, 1, 3, -5, -30],
#           [-30, -5, 3, 1, 1, 3, -5, -30],
#           [-200, -10, -5, 3, 3, -5, -10, -200],
#           [150, 100, -10, -5, -5, -10, 100, 150],
#           [-500, 150, -50, -30, -30, -200, 150, -500]]
# weight = [[1 for i in range(8)] for j in range(8)]

# 找到当前所有合法位置
def get_valid_moves(chessboard, player):
    opponent = -player
    t = time.perf_counter()
    empty_indexes = np.where(chessboard == COLOR_NONE)
    blank_spaces = list(zip(empty_indexes[0], empty_indexes[1]))
    valid_moves = []
    for blank in blank_spaces:
        for dy, dx in directions:
            y = blank[0] + dy
            x = blank[1] + dx
            if 0 <= x < 8 and 0 <= y < 8 and chessboard[y][x] == opponent:
                surround_y = blank[0] + dy
                surround_x = blank[1] + dx
                while 0 <= surround_x < 8 and 0 <= surround_y < 8:
                    if chessboard[surround_y][surround_x] == player:
                        valid_moves.append((blank[0], blank[1]))
                        break
                    elif chessboard[surround_y][surround_x] == COLOR_NONE:
                        break
                    else:
                        surround_y += dy
                        surround_x += dx
    return valid_moves


# 棋子下在当前位置
def action(move, chessboard, current_color):
    chessboard[move[0]][move[1]] = current_color
    flipped_cnt = 0
    for dy, dx in directions:
        r = move[0] + dy
        c = move[1] + dx
        end_with_ally = False
        cnt = 0
        while 0 <= r < 8 and 0 <= c < 8:
            if chessboard[r][c] == COLOR_NONE:
                end_with_ally = False
                break
            elif chessboard[r][c] == -current_color:
                r += dy
                c += dx
                cnt += 1
            else:
                end_with_ally = True
                break

        if end_with_ally:
            flipped_cnt += cnt
            r -= dy
            c -= dx
            for j in range(cnt):
                chessboard[r][c] = current_color
                r -= dy
                c -= dx


def eval_corners(chessboard):

    if chessboard[0][0] != 0:
        # for i, j in corner_position:
        #     weight[i][j] = -1000
        weight[0][1] = -100
        weight[1][0] = -100
        weight[1][1] = -80
        weight[2][2] = -50
        weight[0][2] = -80
        weight[0][3] = -30
        weight[2][0] = -80
        weight[3][0] = -30
    if chessboard[0][7] != 0:
        # for i, j in corner_position:
        #     weight[i][j] = -1000
        weight[0][6] = -100
        weight[1][7] = -100
        weight[1][6] = -80
        weight[2][5] = -50
        weight[0][4] = -30
        weight[0][5] = -80
        weight[2][7] = -80
        weight[3][7] = -30
    if chessboard[7][0] != 0:
        # for i, j in corner_position:
        #     weight[i][j] = -1000
        weight[7][1] = -100
        weight[6][0] = -100
        weight[6][1] = -80
        weight[5][0] = -80
        weight[4][0] = -30
        weight[7][2] = -80
        weight[7][3] = -30
        weight[5][2] = -50
    if chessboard[7][7] != 0:
        # for i, j in corner_position:
        #     weight[i][j] = -1000
        weight[7][6] = -100
        weight[6][7] = -100
        weight[6][6] = -80
        weight[7][5] = -80
        weight[7][4] = -30
        weight[5][7] = -80
        weight[4][7] = -30
        weight[5][5] = -50


def evaluation(chessboard, color):
    score = 0
    his_valid_list = get_valid_moves(chessboard, -color)
    if len(his_valid_list) <= 5:
        score += 50
        for valid in his_valid_list:
            if valid == (0, 0):
                score += 50
            elif valid == (7, 0):
                score += 50
            elif valid == (0, 7):
                score += 50
            elif valid == (7, 7):
                score += 50
    elif len(his_valid_list) < 3:
        score += 200
        for valid in his_valid_list:
            if valid == (0, 0):
                score += 50
            elif valid == (7, 0):
                score += 50
            elif valid == (0, 7):
                score += 50
            elif valid == (7, 7):
                score += 50
    cnt_my = 0
    cnt_opp = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == color:
                score += weight[i][j]
                cnt_my += 1
            elif chessboard[i][j] == -color:
                score -= weight[i][j]
                cnt_opp += 1
    if cnt_my + cnt_opp >= 56:
        score = cnt_opp - cnt_my
    return score, None


def alpha_beta_search(chessboard, color, start_time):
    def max_value(current_chessboard, current_candidate, depth, alpha, beta, current_color, begin_time):
        if depth > MAX_DEPTH:
            return evaluation(current_chessboard, color)
        if time.perf_counter() - begin_time > MAX_TIMEOUT:
            global CUT_BY_TIME
            CUT_BY_TIME = True
            return -infinity, None
        v, move = -infinity, None
        if not current_candidate:
            temp_chessboard = copy.deepcopy(current_chessboard)
            v2, _ = min_value(temp_chessboard, get_valid_moves(temp_chessboard, -current_color), depth + 1, alpha,
                              beta, -current_color, begin_time)
            if v2 > v:
                v, move = v2, None
            alpha = max(alpha, v)
            if alpha >= beta:
                return v, move
        for a in current_candidate:
            temp_chessboard = copy.deepcopy(current_chessboard)
            action(a, temp_chessboard, current_color)
            v2, _ = min_value(temp_chessboard, get_valid_moves(temp_chessboard, -current_color), depth + 1, alpha,
                              beta, -current_color, begin_time)
            if v2 > v:
                v, move = v2, a
            alpha = max(alpha, v)
            if alpha >= beta:
                return v, move
        return v, move

    def min_value(current_chessboard, current_candidate, depth, alpha, beta, current_color, begin_time):
        if depth > MAX_DEPTH:
            return evaluation(current_chessboard, color)
        if time.perf_counter() - begin_time > MAX_TIMEOUT:
            global CUT_BY_TIME
            CUT_BY_TIME = True
            return infinity, None
        v, move = infinity, None
        if not current_candidate:
            temp_chessboard = copy.deepcopy(current_chessboard)
            v2, _ = max_value(temp_chessboard, get_valid_moves(temp_chessboard, -current_color), depth + 1, alpha,
                              beta, -current_color, begin_time)
            if v2 < v:
                v, move = v2, None
            beta = min(beta, v)
            if beta <= alpha:
                return v, move
        for a in current_candidate:
            temp_chessboard = copy.deepcopy(current_chessboard)
            action(a, temp_chessboard, current_color)
            v2, _ = max_value(temp_chessboard, get_valid_moves(temp_chessboard, -current_color), depth + 1, alpha,
                              beta, -current_color, begin_time)
            if v2 < v:
                v, move = v2, a
            beta = min(beta, v)
            if beta <= alpha:
                return v, move
        return v, move

    return max_value(chessboard, get_valid_moves(chessboard, color), 0, -infinity, +infinity, start_time, color)


def define_stage(chessboard):
    global MAX_DEPTH
    rest_cnt = len(np.where(chessboard == COLOR_NONE)[0])
    if 20 <= rest_cnt < 30:
        MAX_DEPTH = 2
    elif 15 <= rest_cnt < 20:
        MAX_DEPTH = 3
    elif 11 <= rest_cnt < 15:
        MAX_DEPTH = 4
    elif 0 < rest_cnt < 11:
        MAX_DEPTH = 5
    else:
        MAX_DEPTH = 2

class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your decision .
        self.candidate_list = []

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        start_time = time.perf_counter()
        self.candidate_list.clear()
        define_stage(chessboard)
        self.candidate_list = get_valid_moves(chessboard, self.color)
        best_value = -10000000
        best_move = None
        for item in self.candidate_list:
            if weight[item[0]][item[1]] > best_value:
                best_move = item
                best_value = weight[item[0]][item[1]]
        if best_move is not None:
            self.candidate_list.append(best_move)

        time_total = 0
        previous_best_move = None

        global MAX_DEPTH, UPPER_DEPTH
        while time_total < MAX_TIMEOUT and MAX_DEPTH <= UPPER_DEPTH:
            value, move = alpha_beta_search(chessboard, self.color, start_time)
            # 被时间掐断的返回结果效果好坏不确定，因此一定使用搜完了产生的结果
            if CUT_BY_TIME:
                if previous_best_move is None:
                    if move is not None:
                        self.candidate_list.append(move)
                else:
                    self.candidate_list.append(previous_best_move)
            else:
                if move is not None:
                    self.candidate_list.append(move)
                    previous_best_move = move
            time_total = time.perf_counter() - start_time
            MAX_DEPTH += 1

        print(MAX_DEPTH - 1)
        return self.candidate_list
