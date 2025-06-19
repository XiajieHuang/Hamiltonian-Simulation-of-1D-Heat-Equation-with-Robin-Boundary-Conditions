from Coordinate_superposition_unitary_Ux import Ux
from Diagonal_block_encoding_of_amplitudes import diagonal_block_encoding_of_amplitudes
from Sparse_Maplitude_Oracle_for_Momentum_Operator import O_Spm_complete, O_Spm_not_banded_matirx
from Banded_Sparse_Access_to_Momentum_Operator import O_BS_A
from fundamental_digital_functions import generate_p_values
from fundamental_gates_functions import dagger, create_name_for_ancilla
from Amplitude_oracle_for_coordinate_operator_Ox import Ox, Ox_general
from QSP_matrix_function_block_encoding import QSP_block_encoding
import numpy as np
from scipy.special import jn
from Shift_Operator import shift_operator
from LCU_auxiliary import lcu_auxiliary


# This file create the Schrodingerisation 1D Heat Equation, it's Hamiltonian is \hat{p}^2 \otimes \eta
def Heat_Equation_Hamiltonian(n_x, n_eta, left_x, right_x, left_eta, right_eta):

    m = 2
    # sparse_indices = [0, 1, 2, 2**n_x - 2]
    # delta_z = (right_x - left_x) / 2 ** n_x # grid size
    # pm_values = [2/(4*delta_z**2), 0 ,-1/(4*delta_z**2), -1/(4*delta_z**2)]
    # Use compact structure
    sparse_indices = [0, 1, 2, 2 ** n_x - 1]
    delta_z = (right_x - left_x) / 2 ** n_x  # grid size
    pm_values = [2 / delta_z ** 2, -1 /  delta_z ** 2, 0, -1 / delta_z ** 2]
    l = 2
    #sparse_indices, pm_values, l = generate_p_values(n_x, m, left_x, right_x)
    o_BS_pm = O_BS_A(n_x, l, sparse_indices, 1)
    N_pm, o_spm = O_Spm_complete(l, m, pm_values)


    # u_eta = Ux(n_eta, left_eta, right_eta)
    # o_eta = diagonal_block_encoding_of_amplitudes(n_eta, u_eta, 0)
    o_eta_total_number_of_qubits, o_eta_ancilla, o_eta = Ox(n_eta)

    # n_extra = 0
    # if o_eta_ancilla > n_x-l:
    #     n_extra = o_eta_ancilla - (n_x-l)
    n_extra = o_eta_ancilla

    gate_sequence = []

    for i in range(1, 1 + l):
        gate_sequence += [{'name': 'h', 'target': [i]}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_spm, 'target': list(range(1 + l))}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_BS_pm,
                       'target': list(range(1, 1 + n_x)) + list(range(1 + n_x + n_extra, 1 + 2*n_x + n_extra))}]
    for i in range(n_x):
        gate_sequence += [{'name': 'swap', 'target': [1 + i, 1 + n_x + n_extra + i]}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': dagger(o_BS_pm),
                       'target': list(range(1, 1 + n_x)) + list(range(1 + n_x + n_extra, 1 + 2*n_x + n_extra))}]
    for i in range(1, 1 + l):
        gate_sequence += [{'name':'h', 'target':[i]}]


    # if o_eta_ancilla > n_x-l:
    #     target_for_o_eta = list(range(1 + l, 1 + n_x + n_extra)) + list(range(1 + 2*n_x + n_extra, 1 + 2*n_x + n_extra + n_eta))
    # else:
    #     target_for_o_eta = list(range(1 + l, 1 + l + o_eta_ancilla)) + list(range(1 + 2*n_x + n_extra, 1 + 2*n_x + n_extra + n_eta))
    target_for_o_eta = list(range(1 + n_x, 1 + n_x + n_extra)) + list(range(1 + 2 * n_x + n_extra, 1 + 2 * n_x + n_extra + n_eta))
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_eta, 'target': target_for_o_eta}]

    number_of_qubits = 1 + 2*n_x + n_extra + n_eta
    number_of_ancilla = 1 + n_x + n_extra
    factor = 2 ** l * N_pm ** (1/2)

    return gate_sequence, factor, number_of_ancilla, number_of_qubits



def Heat_Equation_Hamiltonian_kill_ancilla_version(n_x, n_eta, left_x, right_x, left_eta, right_eta):

    m = 2
    # sparse_indices = [0, 1, 2, 2**n_x - 2]
    # delta_z = (right_x - left_x) / 2 ** n_x # grid size
    # pm_values = [2/(4*delta_z**2), 0 ,-1/(4*delta_z**2), -1/(4*delta_z**2)]
    # Use compact structure
    sparse_indices = [0, 1, 2, 2 ** n_x - 1]
    delta_z = (right_x - left_x) / 2 ** n_x  # grid size
    pm_values = [2 / delta_z ** 2, -1 /  delta_z ** 2, 0, -1 / delta_z ** 2]
    l = 2
    #sparse_indices, pm_values, l = generate_p_values(n_x, m, left_x, right_x)
    o_BS_pm = O_BS_A(n_x, l, sparse_indices, 1)
    N_pm, o_spm = O_Spm_complete(l, m, pm_values)

    o_eta_total_number_of_qubits, o_eta_ancilla, o_eta = Ox(n_eta)
    n_extra = o_eta_ancilla

    number_of_pure_ancilla = n_x - l
    gate_sequence = []
    ancillas_names = []
    # creating ancillas with unique names
    for i in range(number_of_pure_ancilla):
        ancillas_names += [create_name_for_ancilla()]
        gate_sequence += [{"name": "create_ancilla", "parameter": ancillas_names[-1]}]



    for i in range(1, 1 + l):
        gate_sequence += [{'name': 'h', 'target': [i]}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_spm, 'target': list(range(1 + l))}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_BS_pm,
                       'target': list(range(1, 1 + l)) + ancillas_names + list(range(1 + l + n_extra, 1 + 2*n_x + n_extra))}]
    for i in range(l):
        gate_sequence += [{'name': 'swap', 'target': [1 + i, 1 + l + n_extra + i]}]
    for i in range(number_of_pure_ancilla):
        gate_sequence += [{'name': 'swap', 'target': [ancillas_names[i], 1 + 2 * l + n_extra + i]}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': dagger(o_BS_pm),
                       'target': list(range(1, 1 + l)) + ancillas_names + list(range(1 + l + n_extra, 1 + 2*n_x + n_extra))}]
    for i in range(1, 1 + l):
        gate_sequence += [{'name':'h', 'target':[i]}]


    target_for_o_eta = list(range(1 + l, 1 + l + n_extra)) + list(range(1 + l + n_extra + n_x, 1 + l + n_extra + n_x + n_eta))
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_eta, 'target': target_for_o_eta}]

    #Kill ancilla
    for i in range(number_of_pure_ancilla):
        gate_sequence+=[{"name":"kill_ancilla","parameter":ancillas_names[i]}]

    number_of_qubits = 1 + l + n_x + n_extra + n_eta
    number_of_ancilla = 1 + l + n_extra
    factor = 2 ** l * N_pm ** (1/2)

    return gate_sequence, factor, number_of_ancilla, number_of_qubits


def Heat_Equation_Hamiltonian_functional_coefficient(n_x, n_eta, left_x, right_x, L_eta):

    m = 2
    # sparse_indices = [0, 1, 2, 2**n_x - 2]
    # delta_z = (right_x - left_x) / 2 ** n_x # grid size
    # pm_values = [2/(4*delta_z**2), 0 ,-1/(4*delta_z**2), -1/(4*delta_z**2)]
    # Use compact structure
    sparse_indices = [0, 1, 2, 2 ** n_x - 1]
    delta_z = (right_x - left_x) / 2 ** n_x  # grid size
    pm_values = [2 / delta_z ** 2, -1 /  delta_z ** 2, 0, -1 / delta_z ** 2]
    l = 2
    #sparse_indices, pm_values, l = generate_p_values(n_x, m, left_x, right_x)
    o_BS_pm = O_BS_A(n_x, l, sparse_indices, 1)
    N_pm, o_spm = O_Spm_complete(l, m, pm_values)


    o_eta_total_number_of_qubits, o_eta_ancilla, o_eta = Ox(n_eta)


    o_x_number_of_qubits, o_x_ancilla, ox = Ox_general(n_x, left_x, right_x - (right_x - left_x) / 2 ** n_x)
    beta = 0.4
    epsilon = 1e-16
    # Consider the imaginary part beta * sin(t * x), whose L_infinity norm over [-1, 1] is strictly bounded by beta.
    t = 2 * np.pi / (right_x - left_x) * max(abs(left_x), abs(right_x - (right_x - left_x) / 2 ** n_x))  # cos(2pi x/ L)
    parity = 0
    targ = lambda x: beta * np.cos(t * x)

    # Compute the Chebyshev coefficients up to d = 1.4 * |t| + log(1 / epsilon)
    # Such that the approximation error is bounded by epsilon_0
    d = int(np.ceil(1.4 * t + np.log(1 / epsilon)))

    # Generate Chebyshev coefficients
    # x_vals = np.linspace(-1, 1, int(t * 200))  # Sample x values between -1 and 1
    # f_vals = targ(x_vals)  # Evaluate the target function at sample points
    #
    # # Compute Chebyshev coefficients using least-squares fitting
    # coef = np.polynomial.chebyshev.chebfit(x_vals, f_vals, d)
    coef = np.zeros(d + 1)
    coef[0] = jn(0, t) * beta
    for i in range(1, d + 1):
        if i % 2 == 0:
            coef[i] = jn(i, t) * 2 * (-1) ** (i / 2) * beta
    coef[0] += beta  # 1 + cos(2pi x / L)

    opts = {
        'maxiter': 200,
        'criteria': 1e-13,
        'useReal': True,
        'targetPre': True,
        'method': 'FPI'
    }
    poly_x = QSP_block_encoding(True, coef, parity, opts, ox, n_x, o_x_ancilla)
    o_x_number_of_qubits += 1
    o_x_ancilla += 1

    A_pqpm = []
    for i in range(1, 1 + l):
        A_pqpm += [{'name': 'h', 'target': [i]}]
    A_pqpm += [{'name': 'zoom_in', 'block_gate_sequence': o_spm, 'target': list(range(1 + l))}]
    A_pqpm += [{'name': 'zoom_in', 'block_gate_sequence': o_BS_pm,
                       'target': list(range(1, 1 + n_x)) + list(range(1 + n_x  + o_x_ancilla, 1 + 2*n_x  + o_x_ancilla))}]
    for i in range(n_x):
        A_pqpm += [{'name': 'swap', 'target': [1 + i, 1 + n_x + o_x_ancilla + i]}]
    A_pqpm += [{'name': 'zoom_in', 'block_gate_sequence': poly_x,
                       'target': list(range(1 + n_x, 1 + n_x + o_x_ancilla)) + list(
                           range(1 + n_x + o_x_ancilla, 1 + 2 * n_x + o_x_ancilla))}]
    A_pqpm += [{'name': 'zoom_in', 'block_gate_sequence': dagger(o_BS_pm),
                       'target': list(range(1, 1 + n_x)) + list(range(1 + n_x + o_x_ancilla, 1 + 2*n_x + o_x_ancilla))}]
    for i in range(1, 1 + l):
        A_pqpm += [{'name':'h', 'target':[i]}]

    A_pmpq = []
    for i in range(1, 1 + l):
        A_pmpq += [{'name': 'h', 'target': [i]}]
    A_pmpq += [{'name': 'zoom_in', 'block_gate_sequence': o_spm, 'target': list(range(1 + l))}]
    A_pmpq += [{'name': 'zoom_in', 'block_gate_sequence': o_BS_pm,
                'target': list(range(1, 1 + n_x)) + list(
                    range(1 + n_x + o_x_ancilla, 1 + 2 * n_x + o_x_ancilla))}]
    A_pmpq += [{'name': 'zoom_in', 'block_gate_sequence': poly_x,
                'target': list(range(1 + n_x, 1 + n_x + o_x_ancilla)) + list(
                    range(1 + n_x + o_x_ancilla, 1 + 2 * n_x + o_x_ancilla))}]
    for i in range(n_x):
        A_pmpq += [{'name': 'swap', 'target': [1 + i, 1 + n_x + o_x_ancilla + i]}]
    A_pmpq += [{'name': 'zoom_in', 'block_gate_sequence': dagger(o_BS_pm),
                'target': list(range(1, 1 + n_x)) + list(
                    range(1 + n_x + o_x_ancilla, 1 + 2 * n_x + o_x_ancilla))}]
    for i in range(1, 1 + l):
        A_pmpq += [{'name': 'h', 'target': [i]}]

    H1 = [{'name': 'h', 'target': [0]}]
    H1 += [{'name': 'zoom_in', 'block_gate_sequence': A_pqpm, 'target': list(range(1, 2 + 2 * n_x + o_x_ancilla)), 'control': [0], 'control_sequence': [0]}]
    H1 += [{'name': 'zoom_in', 'block_gate_sequence': A_pmpq, 'target': list(range(1, 2 + 2 * n_x + o_x_ancilla)),
            'control': [0], 'control_sequence': [1]}]
    H1 += [{'name': 'h', 'target': [0]}]

    H2 = [{'name': 'rx', 'target': [0], 'parameter': np.pi / 2}]
    H2 += [{'name': 'zoom_in', 'block_gate_sequence': A_pqpm, 'target': list(range(1, 2 + 2 * n_x + o_x_ancilla)),
            'control': [0], 'control_sequence': [1]}]
    H2 += [{'name': 'zoom_in', 'block_gate_sequence': A_pmpq, 'target': list(range(1, 2 + 2 * n_x + o_x_ancilla)),
            'control': [0], 'control_sequence': [0]}]
    H2 += [{'name': 'rx', 'target': [0], 'parameter': np.pi / 2}]
    H2 += [{'name': 'global_phase', 'target': [0], 'parameter': np.pi / 2}]

    # n_extra = max(0, o_eta_ancilla - (n_x - l))
    # H = [{'name': 'h', 'target': [0]}]
    # H += [{'name': 'zoom_in', 'block_gate_sequence': H1,
    #        'target': list(range(1, 3 + n_x)) + list(range(3 + n_x + n_extra, 3 + 2 * n_x + n_extra + o_x_ancilla)),
    #        'control': [0], 'control_sequence': [0]}]
    # H += [{'name': 'zoom_in', 'block_gate_sequence': o_eta,
    #        'target': list(range(3 + l, 3 + l + o_eta_ancilla)) + list(
    #            range(3 + 2 * n_x + n_extra + o_x_ancilla, 3 + 2 * n_x + n_extra + o_x_ancilla + n_eta)),
    #        'control': [0], 'control_sequence': [0]}]
    # H += [{'name': 'zoom_in', 'block_gate_sequence': H2,
    #        'target':list(range(1, 3 + n_x)) + list(range(3 + n_x + n_extra, 3 + 2 * n_x + n_extra + o_x_ancilla)),
    #        'control': [0], 'control_sequence': [1]}]
    # H += [{'name': 'h', 'target': [0]}]
    #
    # number_of_qubits = 3 + 2*n_x + n_extra + o_x_ancilla + n_eta
    # number_of_ancilla = 3 + n_x + n_extra + o_x_ancilla
    normalization_factor = (1 + 1 / L_eta) ** 0.5
    theta = 2 * np.arccos(1 / normalization_factor)

    H = [{'name': 'ry', 'parameter': theta, 'target': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': H1,
           'target': list(range(1, 3 + n_x)) + list(range(3 + n_x + o_eta_ancilla, 3 + 2 * n_x + o_eta_ancilla + o_x_ancilla)),
           'control': [0], 'control_sequence': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': o_eta,
           'target': list(range(3 + n_x, 3 + n_x + o_eta_ancilla)) + list(
               range(3 + 2 * n_x + o_eta_ancilla + o_x_ancilla, 3 + 2 * n_x + o_eta_ancilla + o_x_ancilla + n_eta)),
           'control': [0], 'control_sequence': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': H2,
           'target': list(range(1, 3 + n_x)) + list(range(3 + n_x + o_eta_ancilla, 3 + 2 * n_x + o_eta_ancilla + o_x_ancilla)),
           'control': [0], 'control_sequence': [1]}]
    H += [{'name': 'ry', 'parameter': - theta, 'target': [0]}]

    number_of_qubits = 3 + 2 * n_x + o_eta_ancilla + o_x_ancilla + n_eta
    number_of_ancilla = 3 + n_x + o_eta_ancilla + o_x_ancilla

    factor = 2 ** l * N_pm ** (1/2) / beta * L_eta * normalization_factor ** 2
    # number_of_qubits = 2 + 2 * n_x + o_x_ancilla
    # number_of_ancilla = 2 + n_x + o_x_ancilla
    # factor = 2 ** l * N_pm ** (1/2) / beta

    return H, factor, number_of_ancilla, number_of_qubits


def Heat_Equation_Hamiltonian_functional_coefficient_non_Hermitian(n_x, n_eta, left_x, right_x, left_eta, right_eta):

    m = 2
    # sparse_indices = [0, 1, 2, 2**n_x - 2]
    # delta_z = (right_x - left_x) / 2 ** n_x # grid size
    # pm_values = [2/(4*delta_z**2), 0 ,-1/(4*delta_z**2), -1/(4*delta_z**2)]
    # Use compact structure
    sparse_indices = [0, 1, 2, 2 ** n_x - 1]
    delta_z = (right_x - left_x) / 2 ** n_x  # grid size
    pm_values = [2 / delta_z ** 2, -1 /  delta_z ** 2, 0, -1 / delta_z ** 2]
    l = 2
    #sparse_indices, pm_values, l = generate_p_values(n_x, m, left_x, right_x)
    o_BS_pm = O_BS_A(n_x, l, sparse_indices, 1)
    N_pm, o_spm = O_Spm_complete(l, m, pm_values)


    o_eta_total_number_of_qubits, o_eta_ancilla, o_eta = Ox(n_eta)


    o_x_number_of_qubits, o_x_ancilla, ox = Ox_general(n_x, left_x, right_x - (right_x - left_x) / 2 ** n_x)
    beta = 0.4
    epsilon = 1e-16
    # Consider the imaginary part beta * sin(t * x), whose L_infinity norm over [-1, 1] is strictly bounded by beta.
    t = 2 * np.pi / (right_x - left_x) * max(abs(left_x), abs(right_x - (right_x - left_x) / 2 ** n_x))  # cos(2pi x/ L)
    parity = 0
    targ = lambda x: beta * np.cos(t * x)

    # Compute the Chebyshev coefficients up to d = 1.4 * |t| + log(1 / epsilon)
    # Such that the approximation error is bounded by epsilon_0
    d = int(np.ceil(1.4 * t + np.log(1 / epsilon)))

    # Generate Chebyshev coefficients
    # x_vals = np.linspace(-1, 1, int(t * 200))  # Sample x values between -1 and 1
    # f_vals = targ(x_vals)  # Evaluate the target function at sample points
    #
    # # Compute Chebyshev coefficients using least-squares fitting
    # coef = np.polynomial.chebyshev.chebfit(x_vals, f_vals, d)
    coef = np.zeros(d + 1)
    coef[0] = jn(0, t) * beta
    for i in range(1, d + 1):
        if i % 2 == 0:
            coef[i] = jn(i, t) * 2 * (-1) ** (i / 2) * beta
    coef[0] += beta  # 1 + cos(2pi x / L)

    opts = {
        'maxiter': 200,
        'criteria': 1e-13,
        'useReal': True,
        'targetPre': True,
        'method': 'FPI'
    }
    poly_x = QSP_block_encoding(True, coef, parity, opts, ox, n_x, o_x_ancilla)
    o_x_number_of_qubits += 1
    o_x_ancilla += 1



    gate_sequence = []

    for i in range(1, 1 + l):
        gate_sequence += [{'name': 'h', 'target': [i]}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_spm, 'target': list(range(1 + l))}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_BS_pm,
                       'target': list(range(1, 1 + n_x)) + list(range(1 + n_x  + o_x_ancilla + o_eta_ancilla, 1 + 2*n_x  + o_x_ancilla + o_eta_ancilla))}]
    for i in range(n_x):
        gate_sequence += [{'name': 'swap', 'target': [1 + i, 1 + n_x + o_x_ancilla + o_eta_ancilla + i]}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': poly_x,
                       'target': list(range(1 + n_x, 1 + n_x + o_x_ancilla)) + list(
                           range(1 + n_x + o_x_ancilla + o_eta_ancilla, 1 + 2 * n_x + o_x_ancilla + o_eta_ancilla))}]
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': dagger(o_BS_pm),
                       'target': list(range(1, 1 + n_x)) + list(range(1 + n_x + o_x_ancilla + o_eta_ancilla, 1 + 2*n_x + o_x_ancilla + o_eta_ancilla))}]
    for i in range(1, 1 + l):
        gate_sequence += [{'name':'h', 'target':[i]}]

    target_for_o_eta = list(range(1 + n_x + o_x_ancilla, 1 + n_x + o_x_ancilla + o_eta_ancilla)) + list(range(1 + 2 * n_x + o_x_ancilla + o_eta_ancilla, 1 + 2 * n_x + o_x_ancilla + o_eta_ancilla + n_eta))
    gate_sequence += [{'name': 'zoom_in', 'block_gate_sequence': o_eta, 'target': target_for_o_eta}]

    number_of_qubits = 1 + 2*n_x + o_x_ancilla + o_eta_ancilla + n_eta
    number_of_ancilla = 1 + n_x + o_x_ancilla + o_eta_ancilla
    factor = 2 ** l * N_pm ** (1/2) / beta

    return gate_sequence, factor, number_of_ancilla, number_of_qubits


def Heat_Equation_Hamiltonian_functional_coefficient_new_method(n_x, n_eta, left_x, right_x, L_eta):
    delta_x = (right_x - left_x) / 2 ** n_x
    s_1 = shift_operator(n_x, 1)
    s_minus = shift_operator(n_x, -1)
    auxiliary_for_p2_qubits, auxiliary_for_p2,  auxiliary_for_p2_constant= lcu_auxiliary([-1, 2, -1], False)
    _, auxiliary_for_p2_transpose, _ = lcu_auxiliary([-1, 2, -1], True)

    p2 = [{'name': 'zoom_in', 'block_gate_sequence': auxiliary_for_p2, 'target': list(range(auxiliary_for_p2_qubits))}]
    p2.append({'name': 'zoom_in', 'block_gate_sequence': s_1, 'target': list(range(auxiliary_for_p2_qubits, auxiliary_for_p2_qubits + n_x)), 'control': list(range(auxiliary_for_p2_qubits)), 'control_sequence': [0, 0]})
    p2.append({'name': 'zoom_in', 'block_gate_sequence': s_minus,
               'target': list(range(auxiliary_for_p2_qubits, auxiliary_for_p2_qubits + n_x)),
               'control': list(range(auxiliary_for_p2_qubits)), 'control_sequence': [1, 0]})
    p2.append ({'name': 'zoom_in', 'block_gate_sequence': auxiliary_for_p2_transpose, 'target': list(range(auxiliary_for_p2_qubits))})

    o_eta_total_number_of_qubits, o_eta_ancilla, o_eta = Ox(n_eta)

    o_x_number_of_qubits, o_x_ancilla, o_x = Ox_general(n_x, left_x, right_x - (right_x - left_x) / 2 ** n_x)
    beta = 0.4
    epsilon = 1e-16
    # Consider the imaginary part beta * sin(t * x), whose L_infinity norm over [-1, 1] is strictly bounded by beta.
    t = 2 * np.pi / (right_x - left_x) * max(abs(left_x), abs(right_x - (right_x - left_x) / 2 ** n_x))  # cos(2pi x/ L)
    parity = 0
    targ = lambda x: beta * np.cos(t * x)

    # Compute the Chebyshev coefficients up to d = 1.4 * |t| + log(1 / epsilon)
    # Such that the approximation error is bounded by epsilon_0
    d = int(np.ceil(1.4 * t + np.log(1 / epsilon)))

    # Generate Chebyshev coefficients
    # x_vals = np.linspace(-1, 1, int(t * 200))  # Sample x values between -1 and 1
    # f_vals = targ(x_vals)  # Evaluate the target function at sample points
    #
    # # Compute Chebyshev coefficients using least-squares fitting
    # coef = np.polynomial.chebyshev.chebfit(x_vals, f_vals, d)
    coef = np.zeros(d + 1)
    coef[0] = jn(0, t) * beta
    for i in range(1, d + 1):
        if i % 2 == 0:
            coef[i] = jn(i, t) * 2 * (-1) ** (i / 2) * beta
    coef[0] += beta  # 1 + cos(2pi x / L)

    opts = {
        'maxiter': 200,
        'criteria': 1e-13,
        'useReal': True,
        'targetPre': True,
        'method': 'FPI'
    }
    poly_x = QSP_block_encoding(True, coef, parity, opts, o_x, n_x, o_x_ancilla)
    o_x_number_of_qubits += 1
    o_x_ancilla += 1

    A_pqpm = []
    A_pqpm.append({'name': 'zoom_in', 'block_gate_sequence': p2, 'target': list(range(auxiliary_for_p2_qubits)) + list(range(auxiliary_for_p2_qubits + o_x_ancilla, auxiliary_for_p2_qubits + o_x_ancilla + n_x))})
    A_pqpm.append({'name': 'zoom_in', 'block_gate_sequence': poly_x, 'target': list(range(auxiliary_for_p2_qubits, auxiliary_for_p2_qubits + o_x_ancilla + n_x))})

    A_pmpq = []
    A_pmpq.append({'name': 'zoom_in', 'block_gate_sequence': poly_x,
                   'target': list(range(auxiliary_for_p2_qubits, auxiliary_for_p2_qubits + o_x_ancilla + n_x))})
    A_pmpq.append({'name': 'zoom_in', 'block_gate_sequence': p2, 'target': list(range(auxiliary_for_p2_qubits)) + list(
        range(auxiliary_for_p2_qubits + o_x_ancilla, auxiliary_for_p2_qubits + o_x_ancilla + n_x))})

    number_of_qubits_A_pmpq = auxiliary_for_p2_qubits + o_x_ancilla + n_x
    number_of_ancilla_A_pmpq = auxiliary_for_p2_qubits + o_x_ancilla
    factor_for_A_pmpq = auxiliary_for_p2_constant / delta_x ** 2 / beta

    H1 = [{'name': 'h', 'target': [0]}]
    H1 += [{'name': 'zoom_in', 'block_gate_sequence': A_pqpm, 'target': list(range(1, 1 + auxiliary_for_p2_qubits + o_x_ancilla + n_x)),
            'control': [0], 'control_sequence': [0]}]
    H1 += [{'name': 'zoom_in', 'block_gate_sequence': A_pmpq, 'target': list(range(1, 1 + auxiliary_for_p2_qubits + o_x_ancilla + n_x)),
            'control': [0], 'control_sequence': [1]}]
    H1 += [{'name': 'h', 'target': [0]}]

    # H2 = [{'name': 'rx', 'target': [0], 'parameter': np.pi / 2}]
    H2 = [{'name': 'h', 'target': [0]}]
    H2 += [{'name': 'z', 'target': [0]}]
    H2 += [{'name': 'zoom_in', 'block_gate_sequence': A_pqpm, 'target': list(range(1, 1 + auxiliary_for_p2_qubits + o_x_ancilla + n_x)),
            'control': [0], 'control_sequence': [1]}]
    H2 += [{'name': 'zoom_in', 'block_gate_sequence': A_pmpq, 'target': list(range(1, 1 + auxiliary_for_p2_qubits + o_x_ancilla + n_x)),
            'control': [0], 'control_sequence': [0]}]
    H2 += [{'name': 'h', 'target': [0]}]
    # H2 += [{'name': 'rx', 'target': [0], 'parameter': np.pi / 2}]
    H2 += [{'name': 'global_phase', 'target': [0], 'parameter': np.pi / 2}]

    number_of_qubits_H1 = 1 + number_of_qubits_A_pmpq
    number_of_ancilla_H1 = 1 + number_of_ancilla_A_pmpq
    factor_for_H1 =  factor_for_A_pmpq


    normalization_factor = (1 + 1 / L_eta) ** 0.5
    theta = 2 * np.arccos(1 / normalization_factor)

    H = [{'name': 'ry', 'parameter': theta, 'target': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': H1,
           'target': list(range(1, 1 + number_of_ancilla_H1)) + list(
               range(1 + number_of_ancilla_H1 + o_eta_ancilla, 1 + number_of_ancilla_H1 + o_eta_ancilla + n_x)),
           'control': [0], 'control_sequence': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': o_eta,
           'target': list(range(1 + number_of_ancilla_H1, 1 + number_of_ancilla_H1 + o_eta_ancilla)) + list(
               range(1 + number_of_ancilla_H1 + o_eta_ancilla + n_x, 1 + number_of_ancilla_H1 + o_eta_ancilla + n_x + n_eta)),
           'control': [0], 'control_sequence': [0]}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': H2,
           'target': list(range(1, 1 + number_of_ancilla_H1)) + list(
               range(1 + number_of_ancilla_H1 + o_eta_ancilla, 1 + number_of_ancilla_H1 + o_eta_ancilla + n_x)),
           'control': [0], 'control_sequence': [1]}]
    H += [{'name': 'ry', 'parameter': - theta, 'target': [0]}]

    number_of_qubits_H = 1 + number_of_ancilla_H1 + o_eta_ancilla + n_x + n_eta
    number_of_ancilla_H = 1 + number_of_ancilla_H1 + o_eta_ancilla
    factor_for_H = factor_for_H1 * L_eta * normalization_factor ** 2


    return H, factor_for_H, number_of_ancilla_H, number_of_qubits_H


def Heat_Equation_Hamiltonian_new_method(n_x, n_eta, left_x, right_x, L_eta):
    delta_x = (right_x - left_x) / 2 ** n_x
    s_1 = shift_operator(n_x, 1)
    s_minus = shift_operator(n_x, -1)
    auxiliary_for_p2_qubits, auxiliary_for_p2, auxiliary_for_p2_constant = lcu_auxiliary([-1, 2, -1], False)
    _, auxiliary_for_p2_transpose, _ = lcu_auxiliary([-1, 2, -1], True)

    p2 = [{'name': 'zoom_in', 'block_gate_sequence': auxiliary_for_p2, 'target': list(range(auxiliary_for_p2_qubits))}]
    p2.append({'name': 'zoom_in', 'block_gate_sequence': s_1,
               'target': list(range(auxiliary_for_p2_qubits, auxiliary_for_p2_qubits + n_x)),
               'control': list(range(auxiliary_for_p2_qubits)), 'control_sequence': [0, 0]})
    p2.append({'name': 'zoom_in', 'block_gate_sequence': s_minus,
               'target': list(range(auxiliary_for_p2_qubits, auxiliary_for_p2_qubits + n_x)),
               'control': list(range(auxiliary_for_p2_qubits)), 'control_sequence': [1, 0]})
    p2.append({'name': 'zoom_in', 'block_gate_sequence': auxiliary_for_p2_transpose,
               'target': list(range(auxiliary_for_p2_qubits))})
    p2_factor = auxiliary_for_p2_constant / delta_x ** 2

    o_eta_total_number_of_qubits, o_eta_ancilla, o_eta = Ox(n_eta)

    H = [{'name': 'zoom_in', 'block_gate_sequence': p2,
          'target': list(range(auxiliary_for_p2_qubits)) + list(
              range(auxiliary_for_p2_qubits + o_eta_ancilla, auxiliary_for_p2_qubits + o_eta_ancilla + n_x))}]
    H += [{'name': 'zoom_in', 'block_gate_sequence': o_eta,
           'target': list(range(auxiliary_for_p2_qubits, auxiliary_for_p2_qubits + o_eta_ancilla)) + list(
               range(auxiliary_for_p2_qubits + o_eta_ancilla + n_x,
                     auxiliary_for_p2_qubits + o_eta_ancilla + n_x + n_eta))}]
    H_factor = p2_factor * L_eta
    H_ancilla = auxiliary_for_p2_qubits + o_eta_ancilla
    H_qubit_number = auxiliary_for_p2_qubits + o_eta_ancilla + n_x + n_eta

    return H, H_factor, H_ancilla, H_qubit_number


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





