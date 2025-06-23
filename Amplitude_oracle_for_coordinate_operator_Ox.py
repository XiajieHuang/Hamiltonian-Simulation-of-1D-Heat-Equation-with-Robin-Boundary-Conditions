from auxiliary_unitary_for_new_Ox import Ux
import numpy as np


def Ox(n):
    """
    This function construct a (1, ceil(log2(n)), 0)-block-encoding for diagonal matrix $diag(-1, -1+2/(2^n-1), -1+4/(2^n-1), ..., 1-2/(2^n-1), 1)$,
    it is called coordinate-polynomial-oracle. One is referred to https://arxiv.org/pdf/2411.01131 appendix C  for detailed discussions of this oracle.
    """
    gate_sequence = [] # The gate sequence of this unitary

    number_of_ancilla, ux = Ux(n, False) # number_of_ancilla: The number of ancilla qubits in this block-encoding construction
    _, ux_transpose = Ux(n, True)
    total_number_of_qubits = number_of_ancilla + n # The total number of qubits in this gate

    gate_sequence.append({'name': 'zoom_in', 'block_gate_sequence': ux, 'target': list(range(number_of_ancilla))})

    for i in range(n):
        gate_sequence.append({'name': 'z', 'target': [total_number_of_qubits - 1 - i], 'control': list(range(number_of_ancilla)), 'control_sequence': i})

    gate_sequence.append({'name': 'zoom_in', 'block_gate_sequence': ux_transpose, 'target': list(range(number_of_ancilla))})

    return total_number_of_qubits, number_of_ancilla, gate_sequence


