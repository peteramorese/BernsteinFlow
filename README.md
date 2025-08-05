# BernsteinFlow

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
```
python -m scripts.<script name>
```

## Run via Docker
#### Build and start container
```
cd /path/to/BernsteinFlow
docker compose up -d --build
docker exec -it bernsteinflow-ws-1 /bin/bash
```

#### Run
```
python -m scripts.<script name>
```

## Script directory
The scripts for generating the results in the paper can be found in the `scripts/` directory.
 - benchmark_trajectory_2D_bnf (BNF experiments)
 - benchmark_trajectory_2D_gpgmm (All learned GP-GMM experiments: EKF, WSASOS, GridGMM)
 - benchmark_trajectory_2D_true_gmm (All true model GMM experiments: EKF, WSASOS, GridGMM)
 - density_est_1D (Simple 1D density estimation)
 - density_est_2D (2D density estimation)
 - plot_data (Helper script for making data plots)
 - trajectory_1D (1D trajectory estimation with visualization of the learned/true transition distribution)

The implementation of the algorithms and models used is in the `src/bernstein_flow` directory.