# =======
# This file implements all the functionality needed to manipulate construction blocks

from query_nasbench import NasbenchWrapper
from tqdm import tqdm
from utils import timestamp, calculate_hash
import matplotlib.pyplot as plt
import random

class Data:
	def __init__(self, seed):
		self.seed = seed
		self.nb = NasbenchWrapper(seed)

	def query_batch(self, size = 1000):
		pbar = tqdm(total=size)
		random_cells = {}
		print("I01DT: Collecting unique architectures")
		random.seed(self.seed)
		while len(random_cells) < size:
			#cell_vertices = random.choice([3, 4, 5, 6, 7])
			#cell = self.nb.random_cell(cell_vertices)
			# Lines above makes no sense = there is no way to obtain very low values of accuracy except of some outliers
			# See pdf file for more details
			cell = self.nb.random_cell()
			md5hash = str(calculate_hash(cell))
			if md5hash not in random_cells:
				random_cells[md5hash] = cell
				pbar.update(1)
		pbar.close()
		print("I02DT: Evaluating unique architectures")
		random_results = {}
		for key in tqdm(random_cells):
			random_results[key] = self.nb.query(random_cells[key])
		print("I03DT: Saving log df of observations")
		self.nb.save_df()
		print("I04DT: Plotting observed data")
		return self.nb.df.copy(), random_cells, random_results

	def plot_data(self, df):
		t = timestamp()
		plt.figure()
		plt.title('Distribution of test accuracy in batch (no. architectures {})'.format(len(df.index)))
		plt.hist(df['test_accuracy'], bins=800, density=True)
		#ax = plt.gca()
		#ax.set_xscale('log')
		#ax.set_yscale('log')
		plt.xlabel('Test accuracy')
		plt.grid(True, which="both", ls="-", alpha=0.5)
		plt.tight_layout()
		#plt.xlim(0, 0.3)
		plt.savefig('figures/data_{}test_accuracy_dist'.format(len(df.index)) + str(t) + '.pdf', dpi=600)
		plt.show()

		plt.figure()
		plt.title('Distribution of valiation accuracy in batch (no. architectures {})'.format(len(df.index)))
		plt.hist(df['validation_accuracy'], bins=800, density=True)
		#ax = plt.gca()
		#ax.set_xscale('log')
		#ax.set_yscale('log')
		plt.xlabel('Validation accuracy')
		plt.grid(True, which="both", ls="-", alpha=0.5)
		plt.tight_layout()
		#plt.xlim(0, 0.3)
		plt.savefig('figures/data_{}valid_accuracy_dist'.format(len(df.index)) + str(t) + '.pdf', dpi=600)
		plt.show()
		