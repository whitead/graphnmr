from .gcnmodel import GCNHypers

class GCNHypersStandard(GCNHypers):
    def __init__(self):
        super(GCNHypersStandard, self).__init__()
        self.NUM_EPOCHS = 10000
        self.NUM_BATCHES = 256
        self.BATCH_SIZE = 32
        self.SAVE_PERIOD = 10
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.STRATIFY = False
        self.ATOM_EMBEDDING_SIZE =  256 #Size of space onto which we project elements
        self.EDGE_DISTANCE = True
        self.GCN_RESIDUE = True
        self.DROPOUT_RATE = 0.2 #?

class GCNHypersTiny(GCNHypers):
    def __init__(self):
        super(GCNHypersTiny, self).__init__()
        self.NUM_EPOCHS = 5000
        self.NUM_BATCHES = 256
        self.BATCH_SIZE = 32
        self.SAVE_PERIOD = 10
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.STRATIFY = False
        self.ATOM_EMBEDDING_SIZE =  16
        self.EDGE_EMBEDDING_SIZE =  2
        self.EDGE_EMBEDDING_OUT =  2
        self.EDGE_DISTANCE = True
        self.GCN_RESIDUE = True
        self.DROPOUT_RATE = 0.2
        self.STACKS = 3
        self.LEARNING_RATE = 1e-5
