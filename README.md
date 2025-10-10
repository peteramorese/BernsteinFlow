# SoS RF and BernsteinFlow

You can install dependencies into a virtual environment, or just run a docker container.

## Install dependencies in virtual environment
#### Install
```
cd /path/to/BernsteinFlow
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt 
pip install -e .
```
#### Run
The scripts labeled with "_cross_val" contain the experiments seen in the manuscript. To run a script,
```
python -m scripts.trajectory_6D_sos_cross_val.py
```
and a folder containing all of the files will be generated in the benchmarks directory. The scripts labeled "benchmark_" are used for generating data for the comparison.

The implementation of the SoS Rational Factor form model can be found in src/sos_form/SOSModel.py. The Dynamical systems and their stochastic difference equations can be found in scripts/Systems.py.