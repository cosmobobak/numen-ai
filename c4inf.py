from hyperparams import *
import numpy as np
import pandas as pd
import tensorflow as tf
import os

if __name__ == "__main__":
    loaded_model = tf.keras.models.load_model(
        FINAL_SAVE_PATH + "_" + input(f"enter ID: c4_policy_network_"))

    weights = loaded_model.get_weights()
    single_item_model = CURRENT_ARCH(
        dims=DIMENSIONS, 
        eval_model=False, 
        out_dims=7, 
        xbatch_size=1
    )()

    single_item_model.set_weights(weights)
    # single_item_model.load_weights(CHECKPOINT_PATH)
    single_item_model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    positions = {
        "STARTPOS": [0 for i in range(42)],
        "MIDDLESTACKING": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "WIN_IN_ONE_COL_1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
        "WIN_IN_ONE_2_OR_6": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        "PREVENT_LOSS_ON_COL_6": [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, -1, -1, -1, 0, 0, 1, -1, 1, 1, -1, -1, 1]
    }

    data = [positions[k] for k in positions]

    assert(all([len(p) == 42 for p in data]))

    for d in data:
        print(single_item_model.predict([d])[0])
