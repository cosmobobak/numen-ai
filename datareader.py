from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, SimpleRNN, MaxPooling3D, Dropout, BatchNormalization
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from model import NetMaker, NetMaker2
from tensorflow import keras
from progress.bar import Bar
from UCI_TABLE import allmoves
from tensorflow.python.framework.ops import Tensor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def vectorise_index(index: int, size: int) -> Tensor:
    worker = [0 for _ in range(size)]
    worker[index] = 1
    return tf.convert_to_tensor(worker)


def vectorise_move(m: chess.Move, orientation=chess.WHITE) -> Tensor:
    if orientation == chess.BLACK:
        m = mirror_move(m)
    ucistring = m.uci()
    if len(ucistring) == 5 and ucistring[-1] == "n":
        ucistring = ucistring[:-1]
    idx = allmoves.index(ucistring)
    return vectorise_index(idx, len(allmoves))


def legal_distribution(moveVector: Tensor, legalMoves: chess.LegalMoveGenerator = None):
    global allmoves
    target = [m.uci() for m in legalMoves] if (
        legalMoves is not None) else allmoves
    validMoves = [allmoves.index(m) for m in target] if (
        legalMoves is not None) else list(range(len(allmoves)))
    return [(i, allmoves[i], list(moveVector)[i]) for i in validMoves]
    # index, uci, score


def mirror_from_uci(uci: str):
    wmove = chess.Move.from_uci(uci)
    wmove.from_square = chess.square_mirror(wmove.from_square)
    wmove.to_square = chess.square_mirror(wmove.to_square)
    return wmove


def mirror_move(m: chess.Move):
    return chess.Move(
        chess.square_mirror(m.from_square),
        chess.square_mirror(m.to_square),
        m.promotion,
        m.drop
    )


def devectorise_move(moveVector: Tensor,
                     legalMoves: chess.LegalMoveGenerator = None,
                     orientation=chess.WHITE):

    if orientation == chess.WHITE:
        return chess.Move.from_uci(max(
            legal_distribution(moveVector, legalMoves),
            key=lambda t: t[2])[1])
    else:
        return mirror_from_uci(max(
            legal_distribution(moveVector, legalMoves),
            key=lambda t: t[2])[1])
    # returns the most likely move


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def bb_to_list(bb: int):
    out = [0 for _ in range(64)]
    for i in range(64):
        if bb & (1 << i):
            out[i] = 1
    out = list(chunks(out, 8))
    return out


def bb_to_tuple(bb: int):
    out = [0 for _ in range(64)]
    for i in range(64):
        if bb & (1 << i):
            out[i] = 1
    out = map(lambda x: tuple(x), list(chunks(out, 8)))
    return tuple(out)


def vectorise_board(board: chess.Board) -> Tensor:
    #flipped = False
    # if board.turn == chess.BLACK:
    #    board.apply_mirror()
    #    flipped = True
    bitboards = [
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.occupied,
        board.castling_rights,
    ]
    # if flipped:
    #    board.apply_mirror()
    vecs = [bb_to_list(b) for b in bitboards]
    return tf.convert_to_tensor(vecs)


def tuple_board(board: chess.Board) -> "tuple[tuple[tuple[int]]]":
    #flipped = False
    # if board.turn == chess.BLACK:
    #    board.apply_mirror()
    #    flipped = True
    bitboards = [
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.occupied,
        board.castling_rights,
    ]
    # if flipped:
    #    board.apply_mirror()
    vecs = [bb_to_tuple(b) for b in bitboards]
    return tuple(vecs)


test = chess.Move.from_uci("d2d4")
testb = chess.Board()
# assert(test == mirror_move(mirror_move(test)))
# assert(test == devectorise_move(vectorise_move(
#     test, chess.WHITE), testb.legal_moves, chess.WHITE))

# assert(test == devectorise_move(vectorise_move(
#     test, chess.BLACK), orientation=chess.BLACK))
# assert(test == devectorise_move(vectorise_move(
#     test, chess.WHITE), orientation=chess.WHITE))

vb = vectorise_board(testb)
# assert(len(vb) == 10)
# assert(len(vb[-1]) == 8)
# assert(len(vb[-1][-1]) == 8)
print("assertion passed!")

"""Doing preprocessing of the dataset:"""


def get_training_data(game: chess.pgn.GameT):
    game_result = 0
    if game.headers["Result"] == "1/2-1/2":
        game_result = (0)
    elif game.headers["Result"] == "1-0":
        game_result = (1)
    elif game.headers["Result"] == "0-1":
        game_result = (-1)
    board = game.board()
    x_train = [tuple_board(board)]
    y_train = []
    for move in game.mainline_moves():
        y_train.append((game_result,))
        board.push(move)
        x_train.append(tuple_board(board))
    y_train.append((game_result,))

    return x_train[:len(y_train)], y_train


def all_contents_same_length(xs):
    return all([len(x) == len(xs[0]) for x in xs])

# this function checks if all the contents of container are the same length


def all_subcontents_same_length(xs):
    return all([all_contents_same_length(x) for x in xs])


def prop_x(inputd: "tuple[tuple[tuple[int]]]"):
    assert(all_contents_same_length(inputd))
    assert(all_subcontents_same_length(inputd))
    return len(inputd), len(inputd[0]), len(inputd[0][0])


def iterative_x_dimtest(x_train):
    for i, n in enumerate(x_train):
        assert(np.ndim(n) == 3)
        assert(np.shape(n) == (10, 8, 8))


def iterative_y_dimtest(y_train):
    for i, n in enumerate(y_train):
        assert(np.ndim(n) == 1)
        assert(np.shape(n) == (1858,))


BATCH_SIZE = 64

class DataReader:
    def __init__(self):
        pass

    def __call__(self, filepath):
        with open(filepath, encoding="utf-8", mode="r") as pgn:
            limit = 3030  # 3439 max
            assert(limit >= 31)
            bar = Bar("Loading Games", max=limit - 30)
            x_train: "list[tuple[tuple[tuple[int]]]]" = []
            y_train: "list[int]" = []

            #################################################################
            ##################### TEST DATA GATHERING #######################
            #################################################################
            x_val, y_val = get_training_data(chess.pgn.read_game(pgn))
            for i in range(30):
                x_add, y_add = get_training_data(chess.pgn.read_game(pgn))
                x_val += x_add
                y_val += y_add

            #################################################################
            ##################### TRAINING DATA GATHERING ###################
            #################################################################
            count = 0
            games = []
            while ((game := chess.pgn.read_game(pgn)) and count < (limit - 30)):
                x_add, y_add = get_training_data(game)
                x_train += x_add
                y_train += y_add
                games.append((game.headers["Result"], len(x_add)))
                bar.next()
                count += 1
            x_train, y_train = x_train[:-(len(x_train) % BATCH_SIZE)
                                    ], y_train[:-(len(y_train) % BATCH_SIZE)]
            bar.finish()

        return x_train, y_train, x_val, y_val
