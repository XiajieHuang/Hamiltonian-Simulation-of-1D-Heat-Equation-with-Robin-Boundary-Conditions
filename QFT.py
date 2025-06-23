import numpy as np
from fundamental_gates_functions import dagger


def QFT(n):
    """
    Quantum Fourier Transform
    :param n: Number of qubits
    :return: gate_sequence of QFT
    """
    gate_sequence = []
    for i in range(n):
        gate_sequence += [{'name': 'h', 'target': i}]
        for j in range(i + 1, n):
            gate_sequence += [{'name': 'phase_gate', 'parameter': np.pi/2**(j - i), 'target': i, 'control': [j], 'control_sequence': [1]}]

    if n % 2 == 0:
        for i in range(n//2):
            gate_sequence += [{'name': 'swap', 'target': [i, n - 1 - i]}]
    else:
        for i in range(int(n/2)):
            gate_sequence += [{'name': 'swap', 'target': [i, n - 1 - i]}]
    return gate_sequence


def IQFT(n):
    """
    Inverse Quantum Fourier Transform
    :param n: Number of qubits
    :return: gate_sequence of IQFT
    """
    return dagger(QFT(n))
