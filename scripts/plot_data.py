import numpy as np
import matplotlib.pyplot as plt
import json


ag_dens_est_files = [
    ("BNF", "./benchmarks/trajectory_2D_bnf_2025y_07m_27d_04h_24m_04s/data.json"),
    ("EKF", "./benchmarks/trajectory_2D_ekf_2025y_07m_25d_18h_59m_47s/data.json"),
    ("Grid", "./benchmarks/trajectory_2D_grid_2025y_07m_25d_21h_23m_16s/data.json"),
]


if __name__ == "__main__":

    filenames = ag_dens_est_files

    allh_list = []
    for label, file in filenames:
        with open(file, 'r') as f:
            content = json.load(f)
            allh = content.get("average_log_likelihood", None)
            if allh is None:
                print(f"Field 'average_log_likelihood' not found in {file}")
            allh_list.append((label, allh))
    
    fig, ax = plt.subplots()

    for label, allh_values in allh_list:
        ax.plot(allh_values, label=label)
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Test Log-Likelihood")
    ax.legend()

    plt.show()