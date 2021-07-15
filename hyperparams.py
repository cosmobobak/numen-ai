

BATCH_SIZE = 256
TRAINING_DATA_FILENAME = "50000_15000_onehot.csv"
VALIDATION_DATA_FILENAME = "50000_15000.csv"
CHECKPOINT_PATH = "connect4_checkpoints/cp.ckpt"
FINAL_SAVE_PATH = "C:/github/numen-ai/c4models"
ROWS = 6
COLS = 7
MEMORY_LENGTH = 1
DIMENSIONS = (ROWS, COLS, MEMORY_LENGTH)
EPOCHS = 200
VALIDATION_SPLIT = 0.1

# from model import DoubleConvNetMaker, MainCNN, MLPMaker, DeepConvNetMaker, QuadConvNetMaker
# CURRENT_ARCH = MainCNN
