# MIRACLE (Missing data Imputation Refinement And Causal LEarning)

This is example code for running MIRACLE upon mean imputation.  
This code has been inspired by the work of [1,2,3,4].


## Requirements 

- Python 3.6+
- `tensorflow`
- `numpy`
- `network`
- `scikit-learn`
- `pandas`

## Contents

- `MIRACLE.py` - main regularization file. This code has been forked from [2].
- `run_example.py` - runs a nonlinear toy DAG example.  Uses mean imputation as a baseline and applies MIRACLE to refine. 
- `utils.py` 

## Examples

Base example on toy dag.
```bash
$ python run_example.py
```

An example to run toy example with a dataset size of 2000 for 300 max_steps with a missingness of 30%
```bash
$ python3 run_example.py --dataset_sz 2000 --max_steps 300 --missingness 0.3
```

## References

[1] Jinsung Yoon, James Jordon, and Mihaela van der Schaar. Gain: Missing data imputation using generative adversarial nets. In ICML, 2018.

[2] Trent Kyono, Yao Zhang, and Mihaela van der Schaar. CASTLE: Regularization via auxiliary causal graph discovery. In NeurIPS, 2020.

[3] Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning (NeurIPS 2018). Source code @ https://github.com/xunzheng/notears

[4] Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E. P. (2020). Learning sparse nonparametric DAGs (AISTATS 2020). Source code @ https://github.com/xunzheng/notears