# Deep Implicit Coordination Graphs (DICG)

## Installation
Recommend to run code within a [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) virtual environment. Create a virtual environment by
```
conda create -n dicg python=3.7
```
Activate the virtual environment by running
```
conda activate dicg
```
Install dependencies by running the following command in the root directory of this repo (in the virtual environment):
```
pip install -r requirements.txt
```
The predator-prey environment and the traffic junction environment are included in this repo. To run the StarCraft Multi-agent Challenge (SMAC) environment, please follow the instructions [here](https://github.com/oxwhirl/smac).  You will need to install the full StarCraft II game as well as the SMAC maps.

## Running Experiments
### Predator-Prey
Experiment runners are in `/exp_runners/predatorprey/`, detailed argument specification list can be found in the included runner files (e.g. detailed environment size, number of agents, network sizes, algorithm hyperparemters, etc.). 
To train in Predator-Prey, please `cd` into `/exp_runners/predatorprey/` and run following commands by replacing penalty `P` with desired values (0, -1, -1.25 or -1.5):
| Approach | Command |
|--|--|
| DICG-CE-MLP | `python dicg_ce_predatorprey_runner.py --penalty P` |
| DICG-DE-MLP | `python dicg_de_predatorprey_runner.py --penalty P` |
| CENT-MLP | `python cent_predatorprey_runner.py --penalty P` |
| DEC-MLP | `python dec_predatorprey_runner.py --penalty P` |

The model checkpoints will be saved in `/exp_runners/predatorprey/data/local/`.

Reported **average return** (average of 5 random seeds, utill `7.5e6` environment steps):
| Approach | `P = 0` | `P = -1` | `P = -1.25` | `P = -1.5` |
|--|--|--|--|--|
| DICG-CE-MLP | 78.6 ± 0.1 | 70.2 ± 0.7 | 72.0 ± 1.0 | 72.0 ± 0.7|
| DICG-DE-MLP | 78.2 ± 0.0| 70.5 ± 2.7 | 72.4 ± 1.1 | 70.0 ± 2.7|
| CENT-MLP | 75.0 ± 0.1 | -39.8 ± 3.8 | -52.28 ± 1.3| -65.8 ± 2.9|
| DEC-MLP | 74.4 ± 0.3 | 56.6 ± 0.7 | 56.7 ± 1.1 | 57.9 ± 0.3|

### SMAC
Experiment runners are in `/exp_runners/smac/`, detailed argument specification list can be found in the included runner files. 
To train in SMAC, please `cd` into `/exp_runners/smac/` and run following commands by replacing `MAP` with desired scenario strings (`8m_vs_9m`, `3s_vs_5z` or `6h_vs_8z`):
| Approach | Command |
|--|--|
| DICG-CE-LSTM | `python dicg_ce_smac_runner.py --map MAP` |
| DICG-DE-LSTM | `python dicg_de_smac_runner.py --map MAP` |
| CENT-LSTM | `python cent_smac_runner.py --map MAP` |
| DEC-LSTM | `python dec_smac_runner.py --map MAP` |

The model checkpoints will be saved in `/exp_runners/smac/data/local/`.

Reported **average win rate** (average of 5 random seeds, utill `9e6` environment steps for `8m_vs_9m` and `1.8e7` environment steps for the other two maps):
| Approach | `8m_vs_9m` | `3s_vs_5z` | `6h_vs_8z` |
|--|--|--|--|
|  DICG-CE-LSTM |72 ± 11 %  |96 ± 3 % | 9 ± 9 % |
|  DICG-DE-LSTM |87 ± 6 % |99 ± 1 % | 0 |
|  CENT-LSTM | 42 ± 6 % |0 |0 |
|  DEC-LSTM | 65 ± 16 %  | 94 ± 5 % | 0 |

### Traffic Junction
Experiment runners are in `/exp_runners/traffic/`, detailed argument specification list can be found in the included runner files. 

To train in Traffic Junction, please `cd` into `/exp_runners/traffic/` and run following commands by replacing difficulty `D` with desired setting strings (`easy`, `medium` or `hard`):
| Approach | Command |
|--|--|
| DICG-CE-MLP | `python dicg_ce_traffic_runner.py --difficulty D` |
| DICG-DE-MLP | `python dicg_de_traffic_runner.py --difficulty D` |
| CENT-MLP | `python cent_traffic_runner.py --difficulty D` |
| DEC-MLP | `python dec_traffic_runner.py --difficulty D` |

The model checkpoints will be saved in `/exp_runners/traffic/data/local/`.

Reported **average success rate** (average of 5 random seeds, utill `2.5e6` environment steps):
| Approach | `easy` | `medium` | `hard` |
|--|--|--|--|
|  DICG-CE-MLP |98.1 ± 1.9 % |80.5 ± 6.8 % | 22.8 ± 4.6 % |
|  DICG-DE-MLP |95.6 ± 1.5 % |90.8 ± 2.9 % | 82.2 ± 6.0 % |
|  CENT-MLP | 97.7 ± 0.9 % |0 |0 |
|  DEC-MLP | 90.2 ± 6.5 % | 81.3 ± 4.8 % | 69.4 ± 4.9 % |