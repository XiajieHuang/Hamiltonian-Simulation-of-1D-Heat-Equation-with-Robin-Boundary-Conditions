import numpy as np
from fundamental_digital_functions import binary_form_of_integer


def O_Spm(l, m, pm_values):
    # See paper's formula 15

    if isinstance(pm_values, list):
        pm_values = np.array(pm_values)
    N_pm = np.max(np.abs(pm_values)**2) * 1.001

    thetas = 2 * np.arccos(pm_values/np.sqrt(N_pm)*1j**m).real

    gate_sequence = []
    for i in range(2**l):
        gate_sequence += [{"name": "Ry", "parameter": thetas[i], "target": [0], "control":list(range(1, l+1)),
                           "control_sequence":list(map(int, binary_form_of_integer(i, l)))}]

    return N_pm, gate_sequence


def O_Spm_complete(l, m, pm_values):

    if isinstance(pm_values, list):
        pm_values = np.array(pm_values)
    N_pm = np.max(np.abs(pm_values)**2) * 1.001

    thetas = 2 * np.arccos(pm_values/np.sqrt(N_pm)*1j**m).real

    gate_sequence = []
    for i in range(2**l):
        gate_sequence += [{"name": "Ry", "parameter": thetas[i], "target": [0], "control":list(range(1, l+1)),
                           "control_sequence":list(map(int, binary_form_of_integer(i, l)))}]
    gate_sequence += [{'name': 'global_phase', 'parameter': np.pi*m/2, 'target': [0]}]

    return N_pm, gate_sequence


def O_Spm_not_banded_matirx(l, pm_values, non_circulant_part):

    if isinstance(pm_values, list):
        pm_values = np.array(pm_values)
    if isinstance(non_circulant_part, list):
        non_circulant_part = np.array(non_circulant_part)
    N_pm = np.max(np.abs(np.concatenate((pm_values, non_circulant_part)))**2) * 1.001

    thetas = 2 * np.arccos(pm_values/np.sqrt(N_pm)).real

    gate_sequence = []
    for i in range(2**l):
        gate_sequence += [{"name": "Ry", "parameter": thetas[i], "target": [0], "control":list(range(1, l+1)),
                           "control_sequence":list(map(int, binary_form_of_integer(i, l)))}]

    return N_pm, thetas, gate_sequence