import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from chess import Board, Move, BoardT, WHITE, BLACK
from Viridithas import Viridithas
from datareader import vectorise_board
import tensorflow as tf

class ViridithasNN(Viridithas):
    def __init__(self, human: bool = False, fen: str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', pgn: str = '', timeLimit: int = 15, fun: bool = False, contempt: int = 3000, book: bool = True, advancedTC: list = []):
        super().__init__(human=human, fen=fen, pgn=pgn, timeLimit=timeLimit, fun=fun, contempt=contempt, book=book, advancedTC=advancedTC)
        
        self.model = tf.keras.models.load_model('evalmodel')
        self.model.call = tf.function(
            self.model.call, experimental_relax_shapes=False)
        self.model.summary()
    
    def evaluate(self, depth: float = 1.0) -> float:
        self.nodes += 1
        # sf = self.see_factor() # static exchange eval
        sf = 0
        inputboard = vectorise_board(self.node)

        inputboard = tf.expand_dims(inputboard, axis=0)

        nn = -100 * float(
            self.model(
                inputboard,
                training=False
            )
        )
        return sf + nn

if __name__ == "__main__":
    engine = ViridithasNN(timeLimit=15, book=False, contempt=0)
    #engine.user_setup()
    engine.run_game()
