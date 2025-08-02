import numpy as np
import matplotlib.pyplot as plt
import json


#ag_dens_est_files = [
#    ("BNF", "./benchmarks/trajectory_2D_bnf_2025y_07m_27d_04h_24m_04s/data.json"),
#    ("EKF", "./benchmarks/trajectory_2D_ekf_2025y_07m_25d_18h_59m_47s/data.json"),
#    ("Grid", "./benchmarks/trajectory_2D_grid_2025y_07m_25d_21h_23m_16s/data.json"),
#]

# 12 timestep bnf vs true AG
ag_dens_est_files = [
    ("Grid (learned)", "./benchmarks/trajectory_2D_grid_2025y_07m_25d_21h_23m_16s/data.json", {"linestyle":'-', "color":'red'}),
    ("WSASOS (learned)", "./benchmarks/trajectory_2D_wsasos_2025y_07m_25d_20h_23m_48s/data.json", {"linestyle":'-', "color":'green'}),
    ("EKF (learned)", "./benchmarks/trajectory_2D_ekf_2025y_07m_25d_18h_59m_47s/data.json", {"linestyle":'-', "color":'orange'}),
    ("Grid (true)", "./benchmarks/trajectory_2D_TRUE_grid_2025y_08m_01d_19h_56m_15s/data.json", {"linestyle":'--', "color":'red'}),
    ("WSASOS (true)", "./benchmarks/trajectory_2D_TRUE_wsasos_2025y_08m_01d_19h_58m_24s/data.json", {"linestyle":'--', "color":'green'}),
    ("EKF (true)", "./benchmarks/trajectory_2D_TRUE_ekf_2025y_08m_01d_19h_56m_15s/data.json", {"linestyle":'--', "color":'orange'}),
    ("BNF", "./benchmarks_backup/trajectory_2D_bnf_2025y_08m_01d_21h_23m_27s/data.json", {"linestyle":'-', "color":'blue'}),
]

## 12 timestep bnf vs true NAG
#ag_dens_est_files = [
#    ("Grid (learned)", "./benchmarks/trajectory_2D_grid_2025y_08m_02d_01h_52m_42s/data.json", {"linestyle":'-', "color":'red'}),
#    ("WSASOS (learned)", "./benchmarks/trajectory_2D_wsasos_2025y_08m_02d_04h_02m_42s/data.json", {"linestyle":'-', "color":'green'}),
#    ("EKF (learned)", "./benchmarks/trajectory_2D_ekf_2025y_08m_02d_01h_52m_42s/data.json", {"linestyle":'-', "color":'orange'}),
#    ("BNF", "./benchmarks_backup/trajectory_2D_bnf_2025y_08m_01d_23h_28m_44s/data.json", {"linestyle":'-', "color":'blue'}),
#]





if __name__ == "__main__":

    filenames = ag_dens_est_files

    allh_list = []
    for label, file, settings in filenames:
        with open(file, 'r') as f:
            content = json.load(f)
            allh = content.get("average_log_likelihood", None)
            if allh is None:
                print(f"Field 'average_log_likelihood' not found in {file}")
            allh_list.append((label, allh, settings))
    
    fig, ax = plt.subplots()

    for label, allh_values, settings in allh_list:
        vals = allh_values[:11]
        n_ts = len(vals)
        if label == "WSASOS (learned)":
            ax.plot(range(1, n_ts + 1), vals, label=label, zorder=10, **settings)
        else:
            ax.plot(range(1, n_ts + 1), vals, label=label, **settings)
    
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Average Log-Likelihood", fontsize=13)
    ax.legend()

    fig.savefig("./figures/allh_comparison_nag.pdf")
    plt.show()