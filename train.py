from datareader import DataReader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.framework.ops import Tensor
from UCI_TABLE import allmoves
from progress.bar import Bar
from tensorflow import keras
from model import NetMaker, NetMaker2
from datareader import tuple_board
import tensorflow as tf
import numpy as np
import chess.pgn
import chess
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, SimpleRNN, MaxPooling3D, Dropout, BatchNormalization

BATCH_SIZE = 64
if __name__ == "__main__":
    getter = DataReader()
    x_train, y_train, x_val, y_val = getter(filepath=r"PGNs/Stockfish 10.pgn")

    factory = NetMaker()
    model = factory()

    checkpoint_path = "training_7/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_best_only=True)

    # evalModel.load_weights(checkpoint_path)

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=30,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
        callbacks=[cp_callback]
    )

    model.save("C:/Users/Cosmo/Documents/GitHub/numen-ai/evalmodel")

    # """Evaluate model on test data:"""

    print("Evaluate on test data")
    results = model.evaluate(x_val, y_val, batch_size=128)
    print("test loss, test acc:", results)

    # """Make predictions:"""

    print("Generate predictions for 3 samples")
    boards = [chess.Board(), chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")]
    samples = [tuple_board(b) for b in boards]
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
