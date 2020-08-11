from .gcnmodel import GCNHypers

class GCNHypersStandard(GCNHypers):
    def __init__(self):
        super(GCNHypersStandard, self).__init__()
        self.NUM_EPOCHS = 50
        self.NUM_BATCHES = 256
        self.BATCH_SIZE = 8
        self.SAVE_PERIOD = 1
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.STRATIFY = False
        self.ATOM_EMBEDDING_SIZE =  256 #Size of space onto which we project elements
        self.EDGE_DISTANCE = True
        self.RESIDUE = True
        self.DROPOUT_RATE = 0.0 #?
        self.BATCH_NORM = True
        self.EDGE_FC_LAYERS = 2
        
