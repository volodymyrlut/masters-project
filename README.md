In this project, we introduce the Bayesian Optimization (BO) implementation of the NAS algorithm that is exploiting patterns found in most optimal unique architectures sampled from the most popular NAS dataset and benchmarking tool [NASbench-101](https://github.com/google-research/nasbench). The proposed solution leverages a novel approach to path-encoding and is designed to perform reproducible search even on a relatively small initial batch obtained from the random search. This implementation does not require any special hardware, it is publicly available.

As a side result of this project, we've developed an [WEB-based explanatory tool](volodymyrlut.github.io/masters-project) to explore similar architectures on training time vs test accuracy chart.

To reproduce the results of this work you will need to:

1. Clone this repository
2. Install needed dependencies `pip install -r requirements.txt`
3. Download NASbench 108 epoch file from https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
4. Put this file under `neural-architecture-search/models` folder

Leaving alone results itself, this implementation could be useful for different educational purposes and workshops in the field. That's why we are providing a Jupyter Notebook file `neural_architecture_search/nas-probabilistic.ipynb` for everyone interested in the topic.
