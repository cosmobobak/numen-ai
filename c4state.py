import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from time import time
import numpy as np
from tensorflow.python.keras.engine.training import Model
from hyperparams import *
import pandas as pd
import tensorflow as tf

class Coin:
    def __init__(self):
        self.node = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])

        self.turn = 1
        self.nodes = 0
        self.players = [1, -1]

    def pass_turn(self):
        self.turn = -self.turn

    def is_full(self):  # -> bool //WORKING
        return all((self.node[row][col] != 0 for row in range(6) for col in range(7)))

    def show(self):  # //WORKING
        syms = {
            0: ".",
            1: "X",
            -1: "O",
        }
        mapping = lambda x: syms.get(x, x)
        for row in range(6):
            for col in range(7):
                print(mapping(self.node[row][col]), end=' ')
            print()
        print()

    def legal_moves(self):  # -> std::vector<int>
        return [c for c in [3, 4, 2, 5, 1, 6, 0] if self.node[0][c] == 0]

    def play(self, col):  # //WORKING
        for row in range(6):
            if (self.node[row][col] != 0):
                if (self.turn == 1):
                    self.node[row - 1][col] = self.players[0]
                    break
                else:
                    self.node[row - 1][col] = self.players[1]
                    break

            elif (row == 5):
                if (self.turn == 1):
                    self.node[row][col] = self.players[0]
                else:
                    self.node[row][col] = self.players[1]
        self.turn = -self.turn

    def unplay(self, col):  # //WORKING
        for row in range(6):
            if (self.node[row][col] != 0):
                self.node[row][col] = 0
                break
        self.turn = -self.turn

    def horizontal_term(self):  # -> int
        for row in range(6):
            for col in range(4):
                if (self.node[row][col] == self.node[row][col + 1] and
                        self.node[row][col + 1] == self.node[row][col + 2] and
                        self.node[row][col + 2] == self.node[row][col + 3]):
                    if (self.node[row][col] == self.players[0]):
                        return 1
                    elif (self.node[row][col] == self.players[1]):
                        return -1

    def vertical_term(self):  # -> int
        for row in range(3):
            for col in range(7):
                if (self.node[row][col] == self.node[row + 1][col] and
                        self.node[row + 1][col] == self.node[row + 2][col] and
                        self.node[row + 2][col] == self.node[row + 3][col]):
                    if (self.node[row][col] == self.players[0]):
                        return 1
                    elif (self.node[row][col] == self.players[1]):
                        return -1

    def diagup_term(self):  # -> int
        for row in range(3, 6):
            for col in range(4):
                if (self.node[row][col] == self.node[row - 1][col + 1] and
                        self.node[row - 1][col + 1] == self.node[row - 2][col + 2] and
                        self.node[row - 2][col + 2] == self.node[row - 3][col + 3]):
                    if (self.node[row][col] == self.players[0]):
                        return 1
                    elif (self.node[row][col] == self.players[1]):
                        return -1

    def diagdown_term(self):  # -> int
        for row in range(3):
            for col in range(4):
                if (self.node[row][col] == self.node[row + 1][col + 1] and
                        self.node[row + 1][col + 1] == self.node[row + 2][col + 2] and
                        self.node[row + 2][col + 2] == self.node[row + 3][col + 3]):
                    if (self.node[row][col] == self.players[0]):
                        return 1
                    elif (self.node[row][col] == self.players[1]):
                        return -1

    def evaluate(self):  # -> int
        self.nodes += 1
        v = self.vertical_term()
        if v:
            return v
        h = self.horizontal_term()
        if h:
            return h
        u = self.diagup_term()
        if u:
            return u
        d = self.diagdown_term()
        if d:
            return d

        return 0

    def show_result(self):  # //WORKING
        r = self.evaluate()
        if (r == 0):
            print("1/2-1/2")
        elif (r > 0):
            print("1-0")
        else:
            print("0-1")

    def is_game_over(self):  # -> bool
        return self.evaluate() != 0 or self.is_full()

    def vectorise(self): 
        return np.reshape(self.node, newshape=(42,))

class Agent():
    def __init__(self):
        self.state = Coin()
        loaded_model = tf.keras.models.load_model(
            FINAL_SAVE_PATH + "_" + input(f"enter ID: c4_policy_network_"))

        weights = loaded_model.get_weights()
        self.single_item_model = CURRENT_ARCH(
            dims=DIMENSIONS,
            eval_model=False,
            out_dims=7,
            xbatch_size=1
        )()

        self.single_item_model.set_weights(weights)
        # single_item_model.load_weights(CHECKPOINT_PATH)
        self.single_item_model.compile(
            optimizer=tf.keras.optimizers.SGD(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def play(self):
        vec = self.state.vectorise()
        # print(vec)
        # print(np.ndim(vec))
        move = self.single_item_model.predict(np.array([vec]))[0]
        lmoves = self.state.legal_moves()
        for i in range(7):
            if i not in lmoves:
                move[i] = 0
        self.state.play(np.argmax(move))

def main():
    a = Agent()
    while not a.state.is_game_over():
        a.play()
        a.state.show()
        usrmove = int(input("move\n==> ")) - 1
        a.state.play(usrmove)
        a.state.show()

if __name__ == "__main__":
    main()
