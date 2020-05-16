from nasbench import api
import numpy as np

# =======================
# ===== Not the hyperparameters! Those are the constants from nasbench

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

MAX_VERTICES = 7
MAX_EDGES = 9

# ===== Not the hyperparameters! Those are the constants from nasbench
# =======================


Ops = [INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]
# Create an Inception-like module (5x5 convolution replaced with two 3x3
# convolutions).
Adj = [
    [0, 1, 1, 1, 0, 1, 0],    # input layer
    [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
    [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
    [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
    [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
    [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
    [0, 0, 0, 0, 0, 0, 0]
]

class QueryNasbench:
    def __init__(self):
        # Load the data from file (this will take some time)
        self.nasbench = api.NASBench('./models/nasbench_only108.tfrecord')

    def random_cell(self, adj_matrix_size):
        
        if adj_matrix_size > MAX_VERTICES:
            adj_matrix_size = MAX_VERTICES
            print("W01NB: Max number of vertices reached")

        while True:
            matrix = np.random.choice(
                [0, 1], size=(adj_matrix_size, adj_matrix_size))
            matrix = np.triu(matrix, 1)
            ops = [CONV3X3] * adj_matrix_size
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if self.nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }

    def query(self, Adjacency, Operations = False):
        model_spec = api.ModelSpec(
            # Adjacency matrix of the module
            matrix= Adjacency,   # output layer
            # Operations at the vertices of the module, matches order of matrix
            ops=Operations)

        # Query this model from dataset, returns a dictionary containing the metrics
        # associated with this model.
        data = self.nasbench.query(model_spec)
        fixed_stats, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
        print(fixed_stats)
        return data

    def get_trainable_params(Adjacency, Operations):
        fixed_stats, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
        return fixed_stats['trainable_parameters']

    def get_budget(self):
        return self.nasbench.get_budget_counters()

    def reset_budget(self):
        return self.nasbench.reset_budget_counters()

    def is_valid(self, Adjacency, Operations):
        spec = api.ModelSpec(
            # Adjacency matrix of the module
            matrix = Adjacency,   # output layer
            # Operations at the vertices of the module, matches order of matrix
            ops = Operations
        )
        return self.nasbench.is_valid(spec)


qnb = QueryNasbench()

import pdb; pdb.set_trace()  # breakpoint c076cc0e //

print(qnb.random_cell(3))
