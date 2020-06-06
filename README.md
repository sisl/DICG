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
For exmaple, to train DICG-CE in predator-prey with specific penalty `p = -1.5`, run
```
python /exp_runners/predatorprey/dicg_ce_predatorprey_runner.py --mode train --penalty -1.5
```
The model checkpoints are saved in `/exp_runners/predatorprey/data/local/`.
To evaluate a model, run
```
python /exp_runners/predatorprey/dicg_ce_predatorprey_runner.py --mode eval --exp_name <EXP_NAME>
```
Here `exp_name` is the name of the folder that contains the model checkpoint. It can be omitted if a user specifies the arguments explicitly. The runner can automatically locate the model checkpoint with the specified argument values.

### SMAC
Experiment runners are in `/exp_runners/smac/`, detailed argument specification list can be found in the included runner files. 
The model checkpoints are saved in `/exp_runners/smac/data/local/`.
Training and evaluation commands are similar to those of Predator-Prey.

### Traffic Junction
Experiment runners are in `/exp_runners/traffic/`, detailed argument specification list can be found in the included runner files. 
The model checkpoints are saved in `/exp_runners/traffic/data/local/`.
Training and evaluation commands are similar to those of Predator-Prey.