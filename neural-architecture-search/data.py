# =======
# This file implements all the functionality needed to manipulate construction blocks

from query_nasbench import NasbenchWrapper

class Data()
	def __init__(self, seed):
		self.seed = seed
		self.nb = NasbenchWrapper(seed)

	def query_batch(size):
		for i
		return 

	def plot_data():
		print('Number of architectures', len(test_error) / len(data))

	    plt.figure()
	    plt.title(
	        'Distribution of test error in search space (no. architectures {})'.format(
	            int(len(test_error) / len(data))))
	    plt.hist(test_error, bins=800, density=True)
	    ax = plt.gca()
	    ax.set_xscale('log')
	    ax.set_yscale('log')
	    plt.xlabel('Test error')
	    plt.grid(True, which="both", ls="-", alpha=0.5)
	    plt.tight_layout()
	    plt.xlim(0, 0.3)
	    plt.savefig('nasbench_analysis/search_spaces/export/search_space_1/test_error_distribution.pdf', dpi=600)
	    plt.show()

	    plt.figure()
	    plt.title('Distribution of validation error in search space (no. architectures {})'.format(
	        int(len(valid_error) / len(data))))
	    plt.hist(valid_error, bins=800, density=True)
	    ax = plt.gca()
	    ax.set_xscale('log')
	    ax.set_yscale('log')
	    plt.xlabel('Validation error')
	    plt.grid(True, which="both", ls="-", alpha=0.5)
	    plt.tight_layout()
	    plt.xlim(0, 0.3)
	    plt.savefig('nasbench_analysis/search_spaces/export/search_space_1/valid_error_distribution.pdf', dpi=600)
	    plt.show()

	    print('test_error', min(test_error), 'valid_error', min(valid_error))

