from skopt import gp_minimize
from skopt.space import Integer
from skopt.plots import plot_convergence

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

MAX_VERTICES = 7 # NOT A HYPERPARAMETER; This is a nasbench setting
MAX_EDGES = 9 # NOT A HYPERPARAMETER; This is a nasbench setting

class BOArchitecture:
    
    def __init__(self, probability_matrix, max_time, nb, seed):
        self.seed = seed
        np.random.seed(seed)
        self.max_time = max_time
        self.probas = np.copy(probability_matrix)
        self.nb = nb
        self.search_space_size = 2 * (# 2 - Number of coordinates - row and column
                MAX_EDGES - np.count_nonzero(probability_matrix) # Number of positions left
            )

    
    def f(self, x, eval_model=False):
        rows = x[: : 2]
        columns = x[1: : 2]
        matrix = np.copy(self.probas)
        for c in range(0, len(rows)):
            matrix[rows[c]][columns[c]] = 1
            ops = [CONV3X3] * MAX_VERTICES
            ops[0] = INPUT
            ops[-1] = OUTPUT
            cell = {
                'matrix': np.triu(matrix, 1).tolist(),
                'ops': ops
            }
            reward = 10
            if(self.nb.is_valid(cell)):
                data, adj = self.nb.query(cell)
                if eval_model:
                    return data, adj
                # reward = ((1 - data['validation_accuracy']) / adj.shape[0]) * (1000 / data['training_time'])
                reward = ((1 - data['validation_accuracy']))
            return reward   # Penalty - max 1, min - theoretically 0 but it's unreachable
                                # because we are discounting accuracy by training time which is usually 10x bigger in this dataset

    def optimize_architecture(self):
        self.res = gp_minimize(self.f,       # the function to minimize
            dimensions = [(0, 6)] * int(self.search_space_size),
            noise=0.1,
            acq_func="LCB",                  # the acquisition function
            n_calls=40,                      # the number of evaluations of f
            n_random_starts=30,              # the number of random initialization points
            random_state=self.seed)          # the random seed
        
    def plot_convergence(self):
        plot_convergence(self.res);
        
    def get_best_result(self):
        return self.f(self.res.x, True)