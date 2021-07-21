#from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, SimpleRNN, MaxPooling3D, Dropout, BatchNormalization
import chess
import chess.pgn
import numpy as np
import pandas as pd
from model import ChessNet, BATCH_SIZE
from tensorflow import keras
from progress.bar import Bar
from chessmodels.UCI_TABLE import allmoves
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def vectorise_index(index: int, size: int) -> np.ndarray:
    worker = np.zeros(size)
    worker[index] = 1
    return worker

def vectorise_move(m: chess.Move, orientation=chess.WHITE) -> np.ndarray:
    if orientation == chess.BLACK:
        m = mirror_move(m)
    ucistring = m.uci()
    if len(ucistring) == 5 and ucistring[-1] == "n":
        ucistring = ucistring[:-1]
    idx = allmoves.index(ucistring)
    return vectorise_index(idx, len(allmoves))

def legal_distribution(moveVector: np.ndarray, legalMoves: chess.LegalMoveGenerator = None):
    global allmoves
    target = [m.uci() for m in legalMoves] if (
        legalMoves is not None) else allmoves
    validMoves = [allmoves.index(m) for m in target] if (
        legalMoves is not None) else list(range(len(allmoves)))
    return [(i, allmoves[i], moveVector[i]) for i in validMoves]
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


def devectorise_move(moveVector: np.ndarray,
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

def vectorise_board(board: chess.Board) -> np.ndarray:
    # if board.turn == chess.BLACK:
    #     board = board.mirror()
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
        0xFFFFFFFFFFFFFFFF if (board.turn == chess.WHITE) else 0
    ]
    out = np.zeros((8, 8, 11))
    for i, bb in enumerate(bitboards):
        for square in chess.scan_forward(bb):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            out[file][rank][i] = 1

    return out

"""Doing preprocessing of the dataset:"""
def fen_2_vec(fen: str) -> np.ndarray:
    return vectorise_board(chess.Board(fen=fen))


def mate_2_big(e: str) -> int:
    MATE_SCORE = 1000000
    if e[0] == '#':
        return int(e[1:]) * MATE_SCORE
    return int(e)


BEST_MATERIAL_ADVANTAGE = 10300

def tanh_scale(x: int) -> float:
    return x / BEST_MATERIAL_ADVANTAGE

def get_training_data(maxlen):
    count = 1
    with open("chessmodels/chessData.csv", "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            if count > maxlen:
                break
            fen, preproc_value = line.split(",")
            if "#" in preproc_value:
                continue
            try:
                board, value = fen_2_vec(fen), tanh_scale(int(preproc_value))
            except Exception:
                continue
            # board is an ndarray, value is an int.
            yield board, value
            count += 1
