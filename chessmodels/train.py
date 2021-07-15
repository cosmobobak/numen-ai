import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import ChessNet
from hyperparams import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
from chessmodels.datareader import fen_2_vec, get_training_data

from tensorflow.python.framework.ops import Tensor
from chessmodels.UCI_TABLE import allmoves
from progress.bar import Bar
from tensorflow import keras
import tensorflow as tf
import numpy as np
import chess.pgn
import chess

def main():
    print("getting data!")
    x_train: np.ndarry = get_training_data(usecols=list(range(0, 1)))
    y_train: np.ndarry = get_training_data(usecols=list(range(1, 2)))
    print("data loaded!")

    model = ChessNet(dims=(11, 64), eval_model=True)()

    checkpoint_path = "training_7/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_best_only=True)

    # evalModel.load_weights(checkpoint_path)

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    logdir = os.path.join(os.path.curdir, "chesslogs")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    print("Fit model on training data")
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[cp_callback, es_callback, tb_callback],
        validation_split=VALIDATION_SPLIT
    )

    model.save("C:/Users/Cosmo/Documents/GitHub/numen-ai/evalmodel")
    print("saved model!")

    # """Make predictions:"""

    print("Generate predictions for 3 samples")
    boards = [chess.Board(), chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")]
    samples = [fen_2_vec(b.fen()) for b in boards]
    predictions = model.predict(samples)
    print("predictions shape:", predictions.shape)
    moves_in_order = [sorted(list(enumerate(p)), key=lambda x: x[1])
                      for p in predictions]
    p = [""] * 3
    for i, pred in enumerate(moves_in_order):
        for move in pred:
            if allmoves[move[0]] in list(map(lambda x: x.uci(), boards[i].legal_moves)):
                p[i] = allmoves[move[0]]
                break
    for b in p:
        print(b)

if __name__ == "__main__":
    main()
