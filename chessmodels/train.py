import os

import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import ChessNet, SimpleChessNet
from hyperparams import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
from chessmodels.datareader import BEST_MATERIAL_ADVANTAGE, fen_2_vec, get_training_data

from tensorflow.python.framework.ops import Tensor
from chessmodels.UCI_TABLE import allmoves
from progress.bar import Bar
from tensorflow import keras
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import chess.pgn
import chess

CSV_LINES = 12958036
BATCH_SCALING = 10

def main():
    model = ChessNet()()

    max_examples = 6000000
    maxlen = max_examples // (BATCH_SIZE * BATCH_SCALING) * (BATCH_SIZE * BATCH_SCALING)
    print("getting data!")
    data_generator = get_training_data(maxlen=maxlen)

    count = 0
    x_train, y_train = [], []
    for board, value in tqdm(data_generator, total=maxlen):
        x_train.append(board)
        y_train.append(value)
        count += 1

    assert count == maxlen, f"{count = }, {maxlen = }"
    x_train = np.stack(tuple(x_train))
    y_train = np.array(y_train)
    assert len(x_train) == count
    assert len(y_train) == count

    print("data loaded!")

    print(x_train[0])
    print(y_train[0])

    # exit()

    

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

    model.save("evalmodel")
    print("saved model!")

    # """Make predictions:"""

    print("Generate predictions for 3 samples")
    boards = [chess.Board(), chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")]
    samples = np.array([fen_2_vec(b.fen()) for b in boards])
    predictions = model.predict(samples)
    print(list(map(lambda x: x * BEST_MATERIAL_ADVANTAGE, predictions)))

if __name__ == "__main__":
    main()
