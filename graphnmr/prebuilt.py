from .gcnmodel import GCNHypers

class GCNHypersStandard(GCNHypers):
    def __init__(self):
        super(GCNHypersStandard, self).__init__()
        self.NUM_EPOCHS = 50
        self.NUM_BATCHES = 256
        self.BATCH_SIZE = 16
        self.SAVE_PERIOD = 1
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.STRATIFY = False
        self.ATOM_EMBEDDING_SIZE =  256 #Size of space onto which we project elements
        self.EDGE_DISTANCE = True
        self.RESIDUE = True
        self.DROPOUT_RATE = 0.0 #?
        self.BATCH_NORM = False
        self.EDGE_FC_LAYERS = 3
        self.EDGE_EMBEDDING_SIZE = 32
        

class GCNHypersMedium(GCNHypers):
    def __init__(self):
        super(GCNHypersStandard, self).__init__()
        self.NUM_EPOCHS = 50
        self.NUM_BATCHES = 256
        self.BATCH_SIZE = 16
        self.SAVE_PERIOD = 1
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.STRATIFY = False
        self.ATOM_EMBEDDING_SIZE =  128 #Size of space onto which we project elements
        self.EDGE_DISTANCE = True
        self.RESIDUE = True
        self.DROPOUT_RATE = 0.0 #?
        self.BATCH_NORM = False
        self.EDGE_FC_LAYERS = 2
        self.EDGE_EMBEDDING_SIZE = 8
        self.NON_LINEAR = False
        
        
class GCNHypersSmall(GCNHypers):
    def __init__(self):
        super(GCNHypersStandard, self).__init__()
        self.NUM_EPOCHS = 50
        self.NUM_BATCHES = 256
        self.BATCH_SIZE = 16
        self.SAVE_PERIOD = 1
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.STRATIFY = False
        self.ATOM_EMBEDDING_SIZE =  64 #Size of space onto which we project elements
        self.EDGE_DISTANCE = True
        self.RESIDUE = True
        self.DROPOUT_RATE = 0.0 #?
        self.BATCH_NORM = False
        self.EDGE_FC_LAYERS = 2
        self.EDGE_EMBEDDING_SIZE = 8
        self.EDGE_EMBEDDING_OUT = 2
        self.NON_LINEAR = False

        
class GCNHypersTiny(GCNHypers):
    def __init__(self):
        super(GCNHypersStandard, self).__init__()
        self.NUM_EPOCHS = 50
        self.NUM_BATCHES = 256
        self.BATCH_SIZE = 16
        self.SAVE_PERIOD = 1
        self.EDGE_DISTANCE = True
        self.EDGE_NONBONDED = True
        self.EDGE_LONG_BOND = True
        self.STRATIFY = False
        self.ATOM_EMBEDDING_SIZE =  16 #Size of space onto which we project elements
        self.EDGE_DISTANCE = True
        self.RESIDUE = False
        self.DROPOUT_RATE = 0.0 #?
        self.BATCH_NORM = False
        self.EDGE_FC_LAYERS = 2
        self.EDGE_EMBEDDING_SIZE = 4
        self.STACKS = 3
        self.FC_LAYERS = 2
        self.EDGE_EMBEDDING_OUT = 1
        self.NON_LINEAR = False

        
        
