# Enhanced Causal Effects Estimation Based On Offline Reinforcement Learning

## Summary of the Paper

This paper introduces an enhanced causal effect estimation architecture based on offline reinforcement learning (ECEE-RL), offering adaptive optimization strategies that surpass traditional methods. ECEE-RL models causal effects as a Markov Decision Process, reducing error and sensitivity compared to conventional approaches when applied to simulated datasets and real pilot behavior data.
 
## File Overview
- `d3rlpy/`: This folder contains all python codes of package d3rlpy 1.1.1 version, an Offline Deep Reinforcement Learning Library. The author mainly **made modifications to the following source code scripts.**
  - `d3rlpy/metrics/scorer.py` provides a collection of scoring functions for evaluating reinforcement learning algorithms in our paper.
  - `d3rlpy/algos/torch/ddpg_impl.py` : The `compute_critic_loss` function has been modified based on IBM data.
  - `d3rlpy/algos/torch/sac_impl.py` implements the Soft Actor-Critic (SAC) algorithm in this paper.
  - **Note:** This article compares three baseline methods: Double Machine Learning (DML), Doubly Robust Learners (DRL), and XLearner. This code only showcases the XLearner algorithm, while other algorithms need to be replaced in the corresponding algorithms in the files `scorer.py`, `ddpg_impl.py`, and `sac_impl.py` as mentioned above.

- `data/`: IBM Causal Inference Benchmarking Data provided by the National Vital Statistics System (NVSS) and the National Center for Health Statistics (NCHS)
  - `data_x_y.csv`: Training data
  - `test_data_x_y.csv`: Test data
  - `ibm_x_482_x1.gml`: This is a description of a graph structure to illustrate relationships and dependencies between different attributes in the IBM dataset.

- `Offline-RL_IBM_TD3_Mean_Shuf8debug.py` serves as a platform for implementing the algorithm discussed in our paper, encompassing training, testing, and estimation components.

## Installation Notes
- Install dependencies according to requirements.txt.
- Replace all files in the official `d3rlpy` package directory except for `dataset.cpython-39-xxxxx.so` (varies by system) with the files from this project's `d3rlpy` directory.
- Modify file paths in the scripts:
   - `Offline-RL_IBM_TD3_Mean_Shuf8debug.py` at lines 33 and 40.
   - `d3rlpy/metrics/scorer.py` at lines 17, 18, 216, and 354.
   - `d3rlpy/algos/torch/sac_impl.py` at line 376.

## Citation

Please cite this work appropriately in your research papers and publications.
