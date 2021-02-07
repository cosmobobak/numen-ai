import pickle

class PickleHandler:
    def __init__(self) -> None:
        pass

    def save(self, filename, object):
        outfile = open(filename, 'wb')
        pickle.dump(object, outfile)
        outfile.close()

    def load(self, filename):
        infile = open(filename, 'rb')
        out = pickle.load(infile)
        infile.close()
        return out
