# Enhanced Causal Effects Estimation Based On Offline Reinforcement Learning

## Summary of the Paper

Causal effects estimation is essential for analyzing the causal effects of treatment (intervention) on outcomes. While randomized controlled trials are considered the gold standard for estimating causal effects, conducting them can be challenging or even impossible in some cases. As an alternative, causal effects estimation based on observed data has been widely studied. Traditional causal effects estimation methods, such as causal machine learning, often rely on the strong assumption that there are no unobserved confounding factors. To address this limitation, this paper proposes an enhanced causal effect estimation architecture based on offline reinforcement learning (ECEE-RL).

Unlike fixed strategy traditional methods, ECEE-RL models causal effects estimation as a Markov Decision Process that continuously updates strategies based on feedback. In ECEE-RL, causal effects estimation and sensitivity analysis are treated as "action" and "reward", leading to adaptive optimization of strategies by minimizing the sensitivity of confounders. Experimental results on simulated datasets demonstrate that our method reduces Conditional Average Treatment Effects Mean Squared Error (MSE) by 11.92-22.71% and sensitivity significance by 25.56-64.17% compared to baseline methods under relaxed unconfoundedness. These results indicate that the ECEE-RL architecture offers higher accuracy and better sensitivity while relaxing the assumption of unconfoundedness. The experiments on real data reveal that pilots' control behaviors have varying causal impacts on bioelectrical signals, facial characteristics, and emotions.

<img align="center" src="ECEE-RL architecture.jpg" alt="drawing" width="700">
<div align="center">
  <img src="[https://github.com/******/dbscan_clustering_algorithm/blob/master/data/DBSCAN.png](https://github.com/xiahuan357/ECEE-RL/blob/main/ECEE-RL%20architecture.jpg)">
</div>

**Figure 1:** ECEE-RL Architecture
 
## File Overview
- `d3rlpy/`: Based on the d3rlpy 1.1.1 version, an Offline Deep Reinforcement Learning Library. The author mainly made modifications to the following source code scripts.
  - `d3rlpy/metrics/scorer.py`: 
  - `d3rlpy/algos/torch/ddpg_impl.py`:
  - `d3rlpy/algos/torch/sac_impl.py`:
  - **Note:** This article compares three baseline methods: Double Machine Learning (DML), Doubly Robust Learners (DRL), and XLearner. This code only showcases the XLearner algorithm, while other algorithms need to be replaced in the corresponding algorithms in the files `scorer.py`, `ddpg_impl.py`, and `sac_impl.py` as mentioned above.

- `data/`: IBM Causal Inference Benchmarking Data provided by the National Vital Statistics System (NVSS) and the National Center for Health Statistics (NCHS)
  - `data_x_y.csv`: Training data
  - `test_data_x_y.csv`: Test data
  - `ibm_x_482_x1.gml`:

- `logs/`:
  - `ibm110_SAC_nepos65_ter250_SAMgra_20_scaler0_xleaner_mean_shuffle0_base_real.log`:
  - `ibm110_SAC_nepos65_ter250_SAMgra_20_scaler0_xleaner_mean_shuffle0_rl_predict.log`:

- `tests/`:

- `Offline-RL_IBM_TD3_Mean_Shuf8debug.py`:

- 

## Installation Notes
- Install dependencies according to requirements.txt.
- Replace all files in the official `d3rlpy` package directory except for `dataset.cpython-39-xxxxx.so` (varies by system) with the files from this project's `d3rlpy` directory.
- Modify file paths in the scripts:
   - `Offline-RL_IBM_TD3_Mean_Shuf8debug.py` at lines 33 and 40.
   - `d3rlpy/metrics/scorer.py` at lines 17, 18, 216, and 354.
   - `d3rlpy/algos/torch/sac_impl.py` at line 376.

## Citation

Please cite this work appropriately in your research papers and publications.
