from auxiliary_unitary_for_new_Ox import Ux, Ux_general
import numpy as np


def Ox(n):
    gate_sequence = []

    number_of_ancilla, ux = Ux(n, False)
    _, ux_transpose = Ux(n, True)
    total_number_of_qubits = number_of_ancilla + n

    gate_sequence.append({'name': 'zoom_in', 'block_gate_sequence': ux, 'target': list(range(number_of_ancilla))})

    for i in range(n):
        gate_sequence.append({'name': 'z', 'target': [total_number_of_qubits - 1 - i], 'control': list(range(number_of_ancilla)), 'control_sequence': i})

    gate_sequence.append({'name': 'zoom_in', 'block_gate_sequence': ux_transpose, 'target': list(range(number_of_ancilla))})

    return total_number_of_qubits, number_of_ancilla, gate_sequence


def Ox_general(n, a, b):
    gate_sequence = []

    number_of_ancilla, ux = Ux_general(n, a, b, False)
    _, ux_transpose = Ux_general(n, a, b, True)
    total_number_of_qubits = number_of_ancilla + n

    gate_sequence.append({'name': 'zoom_in', 'block_gate_sequence': ux, 'target': list(range(number_of_ancilla))})

    for i in range(n):
        gate_sequence.append({'name': 'z', 'target': [total_number_of_qubits - 1 - i], 'control': list(range(number_of_ancilla)), 'control_sequence': i})

    gate_sequence.append({'name': 'zoom_in', 'block_gate_sequence': ux_transpose, 'target': list(range(number_of_ancilla))})


    return total_number_of_qubits, number_of_ancilla, gate_sequence