'''romilk_TTS_agent.py
An advanced agent that plays Toro-Tile Straight
romilk@uw.edu
CSE 415
agent name: ibn-e-batoota
'''

from TTS_State import TTS_State
from threading import Event, Thread
import time
stop_event = Event()

USE_CUSTOM_STATIC_EVAL_FUNCTION = False

class MY_TTS_State(TTS_State):
  def static_eval(self):
    if USE_CUSTOM_STATIC_EVAL_FUNCTION:
      return self.custom_static_eval()
    else:
      return self.basic_static_eval()

  def basic_static_eval(self):
    #raise Exception("basic_static_eval not yet implemented.")
    TWF = 0
    TBF = 0
    num_rows = len(self.board)
    num_columns = len(self.board[0])

    for i in range(num_rows):
        for j in range(num_columns):
            if (self.board[i][j] == "W"):

                #case 1 - direct above
                y = y_coord(self, i - 1)
                if self.board[y][j] == " ":
                    TWF = TWF + 1

                #case 2 - direct below
                x = j
                y = y_coord(self, i + 1)
                if self.board[y][x] == " ":
                    TWF = TWF + 1

                #case 3 - towards right
                x = x_coord(self, j - 1)
                y = i
                if self.board[y][x] == " ":
                    TWF = TWF + 1

                #case 4 - towards left
                x = x_coord(self, j + 1)
                y = i
                if self.board[y][x] == " ":
                    TWF = TWF + 1
                
                #case 5 - top left
                x = x_coord(self, j - 1)
                y = y_coord(self, i - 1)
                if self.board[y][x] == " ":
                    TWF = TWF + 1

                #case 6 - top right
                x = x_coord(self, j + 1)
                y = y_coord(self, i - 1)
                if self.board[y][x] == " ":
                    TWF = TWF + 1
                
                #case 7 - bottom left
                x = x_coord(self, j - 1)
                y = y_coord(self, i + 1)
                if self.board[y][x] == " ":
                    TWF = TWF + 1
                
                #case 8 - bottom right
                x = x_coord(self, j + 1)
                y = y_coord(self, i + 1)
                if self.board[y][x] == " ":
                    TWF = TWF + 1

            if (self.board[i][j] == "B"):
                #case 1 - direct above
                x = j
                y = y_coord(self, i - 1)
                if self.board[y][x] == " ":
                    TBF = TBF + 1

                #case 2 - direct below
                x = j
                y = y_coord(self, i + 1)
                if self.board[y][x] == " ":
                    TBF = TBF + 1

                #case 3 - towards right
                x = x_coord(self, j - 1)
                y = i
                if self.board[y][x] == " ":
                    TBF = TBF + 1

                #case 4 - towards left
                x = x_coord(self, j + 1)
                y = i
                if self.board[y][x] == " ":
                    TBF = TBF + 1
                
                #case 5 - top left
                x = x_coord(self, j - 1)
                y = y_coord(self, i - 1)
                if self.board[y][x] == " ":
                    TBF = TBF + 1

                #case 6 - top right
                x = x_coord(self, j + 1)
                y = y_coord(self, i - 1)
                if self.board[y][x] == " ":
                    TBF = TBF + 1
                
                #case 7 - bottom left
                x = x_coord(self, j - 1)
                y = y_coord(self, i + 1)
                if self.board[y][x] == " ":
                    TBF = TBF + 1
                
                #case 8 - bottom right
                x = x_coord(self, j + 1)
                y = y_coord(self, i + 1)
                if self.board[y][x] == " ":
                    TBF = TBF + 1
    result = TWF - TBF
    return result
                        
  def custom_static_eval(self):
    #raise Exception("custom_static_eval not yet implemented.")
    TWF = 0
    TBF = 0
    y_dim = len(self.board)
    x_dim = len(self.board[0])

    for i in range(y_dim):
        for j in range(x_dim):
            if self.board[i][j] == 'W':
                TWF += white_ch(self, i, j)
            if self.board[i][j] == 'B':
                TBF += black_ch(self, i, j)
    return TWF - TBF

def x_coord(self, x):
    x_dim = len(self.board[0])
    return x % x_dim

def y_coord(self, y):
    y_dim = len(self.board)
    return y % y_dim

def white_ch(self, i, j):
    board = self.board
    y_dim = len(board)
    x_dim = len(board[0])
    if board[i][j] == 'W':
        board[i][j] = ' '
        return 10 * white_ch(self, i, (j + 1) % x_dim) * white_ch(self, (i + 1) % y_dim, j)\
            * white_ch(self, i, (j - 1) % x_dim) * white_ch(self, (i - 1) % y_dim, j)\
            * white_ch(self, (i + 1) % y_dim, (j + 1) % x_dim) * white_ch(self, (i - 1) % y_dim, (j + 1) % x_dim)\
            * white_ch(self, (i + 1) % y_dim, (j - 1) % x_dim) * white_ch(self, (i - 1) % y_dim, (j - 1) % x_dim)
    else: 
        return 1

def black_ch(self, i, j):
    board = self.board
    y_dim = len(board)
    x_dim = len(board[0])
    if board[i][j] == 'W':
        board[i][j] = ' '
        return 10 * black_ch(self, i, (j + 1) % x_dim) * black_ch(self, (i + 1) % y_dim, j)\
            * black_ch(self, i, (j - 1) % x_dim) * black_ch(self, (i - 1) % y_dim, j)\
            * black_ch(self, (i + 1) % y_dim, (j + 1) % x_dim) * black_ch(self, (i - 1) % y_dim, (j + 1) % x_dim)\
            * black_ch(self, (i + 1) % y_dim, (j - 1) % x_dim) * black_ch(self, (i - 1) % y_dim, (j - 1) % x_dim)
    else: 
        return 1

# The following is a skeleton for the function called parameterized_minimax,
# which should be a top-level function in each agent file.
# A tester or an autograder may do something like
# import ABC_TTS_agent as player, call get_ready(),
# and then it will be able to call tryout using something like this:
# results = player.parameterized_minimax(**kwargs)

def parameterized_minimax(
       current_state=None,
       max_ply=2,
       use_alpha_beta=False, 
       use_basic_static_eval=False):
    global num_cutoff, num_states_expanded, num_static_eval
    num_cutoff = 0
    num_states_expanded = 0
    num_static_eval = 0
    current_state = MY_TTS_State(current_state.board)
    # All students, add code to replace these default
    # values with correct values from your agent (either here or below).

    provisional = my_minimax(use_alpha_beta, -float("inf"), float("inf"), current_state, use_basic_static_eval, max_ply) 
    DATA = {}
    DATA['CURRENT_STATE_STATIC_VAL'] = provisional
    DATA['N_STATES_EXPANDED'] = num_states_expanded
    DATA['N_STATIC_EVALS'] = num_static_eval
    DATA['N_CUTOFFS'] = num_cutoff

    # STUDENTS: You may create the rest of the body of this function here.

    # Actually return all results...
    return(DATA)


def my_minimax(use_alpha_beta, alpha, beta, current_state, use_basic_static_eval, ply_level):
    # i am not declaring global variables again so let's see
    global num_cutoff
    global num_states_expanded
    global num_static_eval
    num_states_expanded = 1
    num_static_eval = 1
    current_state = MY_TTS_State(current_state.board)
    offsprings = total_offsprings(current_state)
    num_states_expanded += 1
    if ply_level == 0 or len(offsprings) == 0:
        if use_basic_static_eval == False:
            current_value = current_state.static_eval()
        else:
            current_value = current_state.static_eval()
        num_static_eval += 1

        return current_value

    current_player = current_state.whose_turn
    if current_player == "W":
        provisional = -float("inf")
    else:
        provisional = float("inf")
    
    for state in offsprings:
        state = MY_TTS_State(state.board)
        new_value = my_minimax(use_alpha_beta, alpha, beta, state, use_basic_static_eval, ply_level - 1)
        if current_player == "W":
            if new_value > provisional:
                provisional = new_value
            if use_alpha_beta:
                alpha = max(alpha, new_value)
                if alpha >= beta:
                    num_cutoff += 1
                    break
        else:
            if new_value < provisional:
                provisional = new_value
            if use_alpha_beta:
                beta = min(beta, new_value)
                if alpha >= beta:
                    num_cutoff += 1
                    break
    return provisional

def total_offsprings(current_state):
    current_state = MY_TTS_State(current_state.board)
    offsprings = {}
    board = current_state.board
    ver_size = len(board)
    hor_size = len(board[0])
    for i in range(ver_size):
        for j in range(hor_size):
            if board[i][j] == " ":
                new_state = local_move(current_state, (i, j))
                offsprings[new_state] = (i, j)
    return offsprings

def local_move(self, point):
    new_state = self
    turn = self.whose_turn
    if turn == "W":
        new_state.whose_turn = "B"
        new_state.board[point[0]][point[1]] = "B"
    if turn == "B":
        new_state.whose_turn = "W"
        new_state.board[point[0]][point[1]] = "W"

    return new_state

def take_turn(current_state, last_utterance, time_limit):

    # Compute the new state for a move.
    # Start by copying the current state.
    new_state = MY_TTS_State(current_state.board)
    # Fix up whose turn it will be.
    who = current_state.whose_turn
    new_who = 'B'  
    if who=='B': new_who = 'W'  
    new_state.whose_turn = new_who
    global high_term, big_loc, low_term, small_loc
    # Place a new tile
    offsprings = total_offsprings(current_state)

    big_loc = _find_next_vacancy(new_state.board)
    small_loc = big_loc
    state = local_move(current_state, (big_loc))
    high_term = my_minimax(True, -float("inf"), float("inf"), state, False, 2)
    low_term = high_term
    act_time = Thread(target=iddfs, args=(offsprings, stop_event))
    act_time.start()
    act_time.join(timeout=time_limit)
    stop_event.set()

    if new_who == "W":
        location = big_loc
    if new_who == "B":
        location = small_loc

    if location==False: return [[False, current_state], "I don't have any moves!"]
    new_state.board[location[0]][location[1]] = who

    # Construct a representation of the move that goes from the
    # currentState to the newState.
    move = location

    # Make up a new remark
    new_utterance = my_utterance()

    return [[move, new_state], new_utterance]

def iddfs(offsprings, stop_event):
    global high_term, big_loc, low_term, small_loc
    level = 1
    while level <= len(offsprings) and not stop_event.is_set():
        for state in offsprings:
            point = offsprings[state]
            next_state_data = parameterized_minimax(state, level, True, False)
            if next_state_data["CURRENT_STATE_STATIC_VAL"] > high_term:
                big_loc = point
                high_term = next_state_data["CURRENT_STATE_STATIC_VAL"]
            if next_state_data["CURRENT_STATE_STATIC_VAL"] < low_term:
                small_loc = point
                low_term = next_state_data["CURRENT_STATE_STATIC_VAL"]
            level += 1
    

utt_count = 0
def my_utterance():
    global utt_count
    utt_count += 1
    utterance = [
        'Have you ever heard double-bluff ?, now you\'ll know',
        'about to check-mate you!',
        'save your soldiers',
        'brain-work is more important than speed-work!'
        'i\'ll teach you how it\'s done!',
        'get ready to lose',
        'even my donkey can beat you from here',
        'what\'s the antonym of victory ? that\'s what you\'re going to be labeled with',
        'i don\'t want to make you cry, but that\'s your fate',
        'history remembers only conqueres, and here is my historical move',
        'atleast you\'ll learn how winner play'
    ]
    return utterance[utt_count % 11]


def _find_next_vacancy(b):
    for i in range(len(b)):
      for j in range(len(b[0])):
        if b[i][j]==' ': return (i,j)
    return False

def moniker():
    return "ibn-e-batoota" # Return your agent's short nickname here.

def who_am_i():
    return """My name is ibn-e-batoota, created by Romil Kinger.
    This world considers me one of the greatest scholars and explorer.
    I have travelled around the world and learned all the wisdom to win. History remembers me, and
    after todays game, it will be hard for you to forget me."""

def get_ready(initial_state, k, what_side_i_play, opponent_moniker):
    # do any prep, like eval pre-calculation, here.
    global in_initial_state
    in_initial_state = initial_state
    global in_k
    in_k = k
    global in_what_side_i_play
    in_what_side_i_play = what_side_i_play
    global in_opponent_moniker
    in_opponent_moniker = opponent_moniker
    return "OK"