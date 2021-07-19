from model import MainCNN
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from hyperparams import *
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    print(f"{TRAINING_DATA_FILENAME=}\n{VALIDATION_DATA_FILENAME=}")

    # xy_train = pd.read_csv(TRAINING_DATA_FILENAME, usecols=list(range(0, 42 + 7)))

    x_train = pd.read_csv(TRAINING_DATA_FILENAME, usecols=list(range(0, 42)))
    # x_train = xy_train
    print("read x_train!")

    y_train = pd.read_csv(TRAINING_DATA_FILENAME, usecols=list(range(42, 42 + 7)))
    print("read y_train!")

    

    # convert to multiples of BATCH_SIZE
    BATCH_SCALING = 10
    datalen = len(x_train)
    resize_endpoint = datalen // (BATCH_SIZE * BATCH_SCALING) * (BATCH_SIZE * BATCH_SCALING)
    x_train = np.array(x_train[:resize_endpoint])
    y_train = np.array(y_train[:resize_endpoint])
    print(f"trimmed data to chunks of {BATCH_SCALING}x batch_size and converted to ndarrays.")
    print(f"{len(x_train)=}")
    print(f"{len(y_train)=}")

    # y_val = pd.read_csv(VALIDATION_DATA_FILENAME, usecols=[49])

    # get model 
    model = MainCNN(dims=DIMENSIONS, eval_model=False, out_dims=7)()

    model_id = input("Enter current training run ID: \n--> c4_policy_network_")

    # checkpoint stuff
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH + "_" + model_id)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_best_only=True)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

    logdir = os.path.join(os.path.curdir, "logs")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[cp_callback, es_callback, tb_callback],
        validation_split=VALIDATION_SPLIT
    )

    model.save(FINAL_SAVE_PATH + "_" + model_id)

    print(model(np.array([np.array([0 for i in range(42)])])))

if __name__ == "__main__":
    main()
