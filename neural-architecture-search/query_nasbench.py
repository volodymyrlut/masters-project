from nasbench import api
import numpy as np
import pandas as pd
from utils import calculate_hash, timestamp
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


# =======================
# ===== Do not uncomment. This is just for reference of state space

# Ops = [INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]

# Create an Inception-like module (5x5 convolution replaced with two 3x3
# convolutions).

# Adj = [
#     [0, 1, 1, 1, 0, 1, 0],    # input layer
#     [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
#     [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
#     [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
#     [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
#     [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
#     [0, 0, 0, 0, 0, 0, 0]
# ]

# ===== Do not uncomment. This is just for reference of state space
# =======================

class NasbenchWrapper:
    def __init__(self, seed):
        # Set random seed to check reproducibility of results
        self.seed = seed
        np.random.seed()
        # Load the data from file (this will take some time)
        self.nasbench = api.NASBench('./models/nasbench_only108.tfrecord')
        # Lines below are just to construct proper pandas column structure
        cell = self.random_cell()
        model_spec = api.ModelSpec(cell['matrix'], cell['ops'])
        data = self.nasbench.query(model_spec)
        md5hash = calculate_hash(cell)
        data.pop('module_adjacency')
        data.pop('module_operations')
        data['hash'] = md5hash
        self.df = pd.DataFrame.from_records([data], index='hash')
        self.df.drop(self.df.index, inplace=True)
        self.reset_budget() # Clear budgeting of this initial query as this was needed just to capture column names

    def drop_df(self):
        self.df.drop(self.df.index, inplace=True)

    def save_df(self):
        t = timestamp()
        self.df.to_csv("logs/nasbench_wrapper_" + str(t) + ".csv")

    def random_cell(self, adj_matrix_size = MAX_VERTICES):
        
        if adj_matrix_size > MAX_VERTICES:
            # adj_matrix_size = MAX_VERTICES
            print("W01NB: Max number of vertices reached")

        while True:
            matrix = np.random.choice([0, 1], size=(adj_matrix_size, adj_matrix_size))
            matrix = np.triu(matrix, 1)
            ops = [CONV3X3] * adj_matrix_size
            ops[0] = INPUT
            ops[-1] = OUTPUT
            matrix = matrix.tolist()
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if self.nasbench.is_valid(spec):
                # Relying on built-in nasbench functionality to remove
                # useless connection; this is not affecting budjeting
                fixed_stats = self.cut_cell({'matrix': matrix, 'ops': ops})
                m = fixed_stats['module_adjacency'].tolist()
                o = fixed_stats['module_operations']
                if(len(m) == adj_matrix_size):
                    return {
                        'matrix': m,
                        'ops': o
                    }

    def query(self, cell):
        model_spec = api.ModelSpec(**cell)
        md5hash = calculate_hash(cell)
        # Query this model from dataset, returns a dictionary containing the metrics
        # associated with this model.
        data = self.nasbench.query(model_spec)
        adj = data.pop('module_adjacency')
        data.pop('module_operations')
        data['hash'] = md5hash
        onerow = pd.DataFrame.from_records([data], index='hash')
        self.df = pd.concat([self.df, onerow])
        return data, adj

    def cut_cell(self, cell):
        model_spec = api.ModelSpec(**cell)
        fixed_stats, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
        return fixed_stats

    def get_trainable_params(self, cell):
        model_spec = api.ModelSpec(**cell)
        fixed_stats, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
        return fixed_stats['trainable_parameters']

    def get_budget(self):
        return self.nasbench.get_budget_counters()

    def reset_budget(self):
        return self.nasbench.reset_budget_counters()

    def is_valid(self, cell):
        spec = api.ModelSpec(**cell)
        return self.nasbench.is_valid(spec)
