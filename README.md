# Enhanced Causal Effects Estimation Based On Offline Reinforcement Learning

## Summary of the Paper

Causal effects estimation is essential for analyzing the causal effects of treatment (intervention) on outcomes. While randomized controlled trials are considered the gold standard for estimating causal effects, conducting them can be challenging or even impossible in some cases. As an alternative, causal effects estimation based on observed data has been widely studied. Traditional causal effects estimation methods, such as causal machine learning, often rely on the strong assumption that there are no unobserved confounding factors. To address this limitation, this paper proposes an enhanced causal effect estimation architecture based on offline reinforcement learning (ECEE-RL).

Unlike fixed strategy traditional methods, ECEE-RL models causal effects estimation as a Markov Decision Process that continuously updates strategies based on feedback. In ECEE-RL, causal effects estimation and sensitivity analysis are treated as "action" and "reward", leading to adaptive optimization of strategies by minimizing the sensitivity of confounders. Experimental results on simulated datasets demonstrate that our method reduces Conditional Average Treatment Effects Mean Squared Error (MSE) by 11.92-22.71% and sensitivity significance by 25.56-64.17% compared to baseline methods under relaxed unconfoundedness. These results indicate that the ECEE-RL architecture offers higher accuracy and better sensitivity while relaxing the assumption of unconfoundedness. The experiments on real data reveal that pilots' control behaviors have varying causal impacts on bioelectrical signals, facial characteristics, and emotions.

<p align="center">
  <img src="https://github.com/xiahuan357/ECEE-RL/blob/main/ECEE-RL%20Achitecture.jpg" alt="drawing" width="700">
</p>

<p align="center"><b>Figure 1:</b> ECEE-RL Architecture</p>

<p align="center">
  <img src="https://github.com/xiahuan357/ECEE-RL/blob/main/Baseline%20Methods.jpg" alt="drawing" width="900">
</p>

<p align="center"><b>Figure 2:</b> Three baseline methods CATE and p-values compared to those obtained with ECEE-RL</p>
 
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

- `logs/`: This folder contains a collection of information outputted by a logger during experimental processes.

- `tests/`: This folder contains all datasets, Python codes, and results related to the real data from the Human-Machine data of the pilot simulating the flight process on the aircraft simulator in this paper.

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
