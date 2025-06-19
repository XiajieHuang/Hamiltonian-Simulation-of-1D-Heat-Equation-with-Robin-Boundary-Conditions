from Sparse_Maplitude_Oracle_for_Momentum_Operator import O_Spm_complete, O_Spm_not_banded_matirx
from Banded_Sparse_Access_to_Momentum_Operator import O_BS_A
from fundamental_gates_functions import dagger, create_name_for_ancilla
from Amplitude_oracle_for_coordinate_operator_Ox import Ox, Ox_general
import numpy as np


def Heat_Equation_Hamiltonain_Robin(n_x, n_eta, left_x, right_x, L_eta, alpha1, alpha2):
    m = 2
    # sparse_indices = [0, 1, 2, 2**n_x - 2]
    # delta_z = (right_x - left_x) / 2 ** n_x # grid size
    # pm_values = [2/(4*delta_z**2), 0 ,-1/(4*delta_z**2), -1/(4*delta_z**2)]
    # Use compact structure
    sparse_indices = [0, 1, 2, 3, 4, 5, 2 ** n_x - 2, 2 ** n_x - 1]
    delta_z = (right_x - left_x) / (2 ** n_x - 1)  # grid size
    pm_values = [-5/2 / delta_z ** 2, 4/3 / delta_z ** 2, -1/12 / delta_z ** 2, 0, 0, 0, -1/12 / delta_z ** 2, 4/3 / delta_z ** 2]
    l = 3
    # sparse_indices, pm_values, l = generate_p_values(n_x, m, left_x, right_x)
    o_BS_pm = O_BS_A(n_x, l, sparse_indices, 1)
    non_circulant_part = [7*alpha1*delta_z/3 - 5/2, 8/3, -1/6, 4/3 - alpha1*delta_z/6, -31/12, 4/3+alpha2*delta_z/6, -5/2-7*alpha2*delta_z/3]
    non_circulant_part = [x/delta_z ** 2 for x in non_circulant_part]
    N_pm, thetas, o_spm = O_Spm_not_banded_matirx(l, pm_values, non_circulant_part)

    A = []
    for i in range(1, 1 + l):
        A += [{'name': 'h', 'target': [i]}]
    A += [{'name': 'zoom_in', 'block_gate_sequence': o_spm, 'target': list(range(1 + l))}]
    A += [{'name': 'zoom_in', 'block_gate_sequence': o_BS_pm,
                       'target': list(range(1, 1 + 2*n_x))}]
    for i in range(n_x):
        A += [{'name': 'swap', 'target': [1 + i, 1 + n_x + i]}]

    #Fix the non_banded rows
    # The 0th row
    theta = 2 * np.arccos((7*alpha1*delta_z/3 - 5/2) / delta_z ** 2 /np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[0], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * (2 * n_x)}]
    theta = 2 * np.arccos(8/3 / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[1], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * (n_x - 1) + [1] + [0] * n_x}]
    theta = 2 * np.arccos(-1/6 / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[2], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * (n_x - 2) + [1] + [0] * (n_x + 1)}]
    theta = 2 * np.arccos(0).real
    A += [{'name': 'ry', 'parameter': theta - thetas[-2], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * (n_x - 1) + [0] * (n_x + 1)}]
    theta = 2 * np.arccos(0).real
    A += [{'name': 'ry', 'parameter': theta - thetas[-1], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * n_x + [0] * n_x}]
    # The first row
    theta = 2 * np.arccos((4/3 - alpha1*delta_z/6) / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[-1], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * (2 * n_x - 1) + [1]}]
    theta = 2 * np.arccos(-31/12 / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[0], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * (n_x - 1) + [1] + [0] * (n_x - 1) + [1]}]
    theta = 2 * np.arccos(0).real
    A += [{'name': 'ry', 'parameter': theta - thetas[-2], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * n_x + [0] * (n_x - 1) + [1]}]
    # The (n_x - 1)th row
    theta = 2 * np.arccos(0).real
    A += [{'name': 'ry', 'parameter': theta - thetas[2], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * n_x + [1] * (n_x - 1) + [0]}]
    theta = 2 * np.arccos(-31/12 / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[0], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * (n_x - 1) + [0] + [1] * (n_x - 1) + [0]}]
    theta = 2 * np.arccos((4/3+alpha2*delta_z/6) / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[1], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * n_x + [1] * (n_x - 1) + [0]}]
    # The (n_x)th row
    theta = 2 * np.arccos(0).real
    A += [{'name': 'ry', 'parameter': theta - thetas[1], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * n_x + [1] * n_x}]
    theta = 2 * np.arccos(0).real
    A += [{'name': 'ry', 'parameter': theta - thetas[2], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [0] * (n_x - 1) + [1] + [1] * n_x}]
    theta = 2 * np.arccos(-1/6 / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[-2], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * (n_x - 2) + [0] + [1] + [1] * n_x}]
    theta = 2 * np.arccos(8/3 / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[-1], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * (n_x - 1) + [0] + [1] * n_x}]
    theta = 2 * np.arccos((-5/2-7*alpha2*delta_z/3) / delta_z ** 2 / np.sqrt(N_pm)).real
    A += [{'name': 'ry', 'parameter': theta - thetas[0], 'target': [0], 'control': list(range(1, 1 + 2 * n_x)),
           'control_sequence': [1] * n_x + [1] * n_x}]


    A += [{'name': 'zoom_in', 'block_gate_sequence': dagger(o_BS_pm),
                       'target': list(range(1, 1 + 2*n_x))}]
    for i in range(1, 1 + l):
        A += [{'name':'h', 'target':[i]}]

    A_factor = 2 ** l * N_pm ** (1/2)
    A_ancilla = 1 + n_x
    A_qubit_number = 1 + 2 * n_x

    def control_ua_phi(ua, number_of_ancilla_ua, number_of_register_ua, factor_ua, phi):
        control_ua_block_encoding = []
        control_ua_block_encoding.append({'name': 'zoom_in', 'block_gate_sequence': ua,
                                          'target': list(range(number_of_ancilla_ua)) + list(
                                              range(number_of_ancilla_ua + 1,
                                                    number_of_ancilla_ua + 1 + number_of_register_ua)),
                                          'control': [number_of_ancilla_ua], 'control_sequence': [0]})
        control_ua_block_encoding.append(
            {'name': 'global_phase', 'parameter': phi, 'target': [0], 'control': [number_of_ancilla_ua],
             'control_sequence': [1]})
        control_ua_factor = factor_ua
        control_ua_ancilla = number_of_ancilla_ua
        control_ua_qubit_number = number_of_ancilla_ua + number_of_register_ua + 1
        return control_ua_block_encoding, control_ua_factor, control_ua_ancilla, control_ua_qubit_number


    control_ua_block_encoding_0, control_ua_factor, control_ua_ancilla, control_ua_qubit_number = control_ua_phi(A, A_ancilla, n_x, A_factor, 0)
    control_ua_dagger_block_encoding_pi, _, _, _ = control_ua_phi(dagger(A), A_ancilla, n_x, A_factor, np.pi)


    A_plus_A_dagger_over_2 = []
    A_plus_A_dagger_over_2.append({'name': 'h', 'target': [0]})
    A_plus_A_dagger_over_2.append({'name': 'zoom_in', 'block_gate_sequence': control_ua_block_encoding_0,
                                   'target': list(range(1, 1 + control_ua_qubit_number)), 'control': [0],
                                   'control_sequence': [0]})
    A_plus_A_dagger_over_2.append({'name': 'zoom_in', 'block_gate_sequence': control_ua_dagger_block_encoding_pi,
                                   'target': list(range(1, 1 + control_ua_qubit_number)), 'control': [0],
                                   'control_sequence': [1]})
    A_plus_A_dagger_over_2.append({'name': 'h', 'target': [0]})

    A_plus_A_dagger_over_2_factor = control_ua_factor
    A_plus_A_dagger_over_2_ancilla = control_ua_ancilla + 1
    A_plus_A_dagger_over_2_qubit_number = control_ua_qubit_number + 1

    #S1
    S1_factor = A_plus_A_dagger_over_2_factor + 1 / 2
    S1_ancilla = A_plus_A_dagger_over_2_ancilla + 1
    S1_qubit_number = A_plus_A_dagger_over_2_qubit_number + 1
    theta = 2*np.arccos(np.sqrt(A_plus_A_dagger_over_2_factor) / np.sqrt(S1_factor))
    S1 = [{'name': 'ry', 'parameter': theta, 'target': [0]}]
    S1 += [{'name': 'zoom_in', 'block_gate_sequence': A_plus_A_dagger_over_2, 'target': list(range(1, S1_qubit_number)),
            'control': [0], 'control_sequence': [0]}]
    S1 += [{'name': 'x', 'target': [2 + A_ancilla], 'control': [0], 'control_sequence': [1]}]
    S1 += [{'name': 'ry', 'parameter': -theta, 'target': [0]}]


    #S2
    control_ua_dagger_block_encoding_0, _, _, _ = control_ua_phi(dagger(A), A_ancilla, n_x, A_factor, 0)
    A_minus_A_dagger_over_2j = []
    A_minus_A_dagger_over_2j.append({'name': 'h', 'target': [0]})
    A_minus_A_dagger_over_2j.append({'name': 'z', 'target': [0]})
    A_minus_A_dagger_over_2j.append({'name': 'global_phase', 'parameter': - np.pi / 2, 'target': [0]})
    A_minus_A_dagger_over_2j.append({'name': 'zoom_in', 'block_gate_sequence': control_ua_block_encoding_0,
                                   'target': list(range(1, 1 + control_ua_qubit_number)), 'control': [0],
                                   'control_sequence': [0]})
    A_minus_A_dagger_over_2j.append({'name': 'zoom_in', 'block_gate_sequence': control_ua_dagger_block_encoding_0,
                                   'target': list(range(1, 1 + control_ua_qubit_number)), 'control': [0],
                                   'control_sequence': [1]})
    A_minus_A_dagger_over_2j.append({'name': 'h', 'target': [0]})

    S2 = [{'name': 'ry', 'parameter': theta, 'target': [0]}]
    S2 += [{'name': 'zoom_in', 'block_gate_sequence': A_minus_A_dagger_over_2j, 'target': list(range(1, S1_qubit_number)),
            'control': [0], 'control_sequence': [0]}]
    S2 += [{'name': 'y', 'target': [2 + A_ancilla], 'control': [0], 'control_sequence': [1]}]
    S2 += [{'name': 'ry', 'parameter': -theta, 'target': [0]}]


    # Add eta dimension
    o_eta_total_number_of_qubits, o_eta_ancilla, o_eta = Ox(n_eta)
    H_factor = S1_factor * (1 + L_eta)
    H_ancilla = 1 + S1_ancilla + o_eta_ancilla
    H_qubit_number = 1 + S1_qubit_number + o_eta_total_number_of_qubits
    theta = 2 * np.arccos(np.sqrt(S1_factor * L_eta) / np.sqrt(H_factor))
    H = [{'name': 'ry', 'parameter': theta, 'target': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': S1, 'target': list(range(1, 1 + S1_ancilla)) + list(
        range(1 + S1_ancilla + o_eta_ancilla, 1 + S1_qubit_number + o_eta_ancilla)), 'control': [0],
           'control_sequence': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': o_eta, 'target': list(range(1 + S1_ancilla, 1 + S1_ancilla + o_eta_ancilla)) + list(
        range(1 + S1_qubit_number + o_eta_ancilla, 1 + S1_qubit_number + o_eta_ancilla + n_eta)), 'control': [0],
           'control_sequence': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': S2, 'target': list(range(1, 1 + S1_ancilla)) + list(
        range(1 + S1_ancilla + o_eta_ancilla, 1 + S1_qubit_number + o_eta_ancilla)), 'control': [0],
           'control_sequence': [1]}]
    H += [{'name': 'ry', 'parameter': -theta, 'target': [0]}]


    return H, H_factor, H_ancilla, H_qubit_number





