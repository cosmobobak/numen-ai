import numpy as np
from model import SimpleChessNet, ChessNet
import chess
import chess.pgn
import tensorflow as tf
from chessmodels.datareader import fen_2_vec

#loaded_model = tf.keras.models.load_model(
#    "C:/Users/Cosmo/Documents/GitHub/numen-ai/evalmodel")

#weights = loaded_model.get_weights()
single_item_model = SimpleChessNet(
    xbatch_size=1
)()

#single_item_model.set_weights(weights)
single_item_model.load_weights("training_7/cp.ckpt")
single_item_model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)

board = chess.Board()

while not board.is_game_over():
    print(board)
    best = float("-inf")
    best_move = chess.Move.null()
    for move in board.generate_legal_moves():
        board.push(move)
        tgt = np.array([fen_2_vec(board.fen())])
        value: float = single_item_model(tgt)[0]
        board.pop()
        if board.turn == chess.BLACK:
            value = -value
        if value > best:
            best = value
            best_move = move
    board.push(best_move)

print(chess.pgn.Game.from_board(board)[-1])
