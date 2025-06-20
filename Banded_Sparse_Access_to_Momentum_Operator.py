import numpy as np
from fundamental_gates_functions import create_name_for_ancilla
from fundamental_digital_functions import binary_form_of_integer

def O_BS_A(n_qubits, l, sparse_indices, order):
    """
    This function construct the O_BS_A oracle, see https://arxiv.org/pdf/2405.12855 Lemma 1 for more details.

    parameters:
    n_qubits: the number of the main register qubits
    l: the number of the sparse register qubits
    sparse_indices: array of sparse indices in the firs row
    order == 0 : |0>|s>|i> --> |r_{si}>|i>,   order == 1 : |s>|0>|i> --> |r_{si}>|i>
    """

    u_la = U_lA(n_qubits, l, sparse_indices, order)
    u_sum = U_SUM(n_qubits, 0)

    gate_sequence = [{"name": "zoom_in", "block_gate_sequence": u_la, "target": list(range(n_qubits)), "control": [], "control_sequence": []}]
    gate_sequence += [
        {"name": "zoom_in", "block_gate_sequence": u_sum, "target": list(range(2 * n_qubits)),
         "control": [], "control_sequence": []}]

    return gate_sequence

def U_SUM(n_qubits, target_register):
    """
    This function construct the U_SUM gate in https://arxiv.org/pdf/2405.12855 appendix A Eq.(A2), this oracle
    implements addition operation, it transforms state |i>|j> to |i+j modulo 2>|j>.


    parameters:
    n_qubits: the number of the main register qubits
    target_register: controls where to write the answer
    if target_register==1:
    U_SUM|i>|j>=U_SUM|i>|j+i modulo 2>
    if target_register==0:
    U_SUM|i>|j>=U_SUM|i+j modulo 2>|j>


    The Circuit requires 3n-1 qubits

    n_qubits first register
    n_qubits  second register
    n_qubits-1 pure ancillas
    """
    gate_sequence = []

    """
    The quantum scheme requires n-1 ancillas for those pure ancillas I create the separate notation
    a0 -- pure ancilla number one
    a1 -- pure ancilla number two
    ...

    so, if the target/control qubit: 1 -- indicates that we the first qubit of the use main register
    a0 -- the zero's pure ancilla

    WARNING to not mess pure ancillas with other qubit
    at the end of the function they are  |0>

    """
    ancillas_names=[]
    #creating ancillas with unique names
    for i in range(n_qubits-1):
        ancillas_names+=[create_name_for_ancilla()]
        gate_sequence+=[{"name":"create_ancilla","parameter":ancillas_names[-1]}]
    
    """
    Gate sequence is base on the Fig.10
    """
    if target_register == 1:  # U_SUM|i>|j>=U_SUM|i>|j+i modulo 2>

        # addition of the least qubits
        gate_sequence += [{"name": "X", "control": [n_qubits - 1, 2 * n_qubits - 1], "target": [ancillas_names[0]], "control_sequence": [1, 1]}]
        gate_sequence += [{"name": "X", "control": [n_qubits - 1], "target": [2 * n_qubits - 1], "control_sequence": [1]}]

        # the forward pass
        for i in range(1, n_qubits - 1):
            gate_sequence += [
                {"name": "X", "control": [n_qubits - 1 - i, 2 * n_qubits - 1 - i], "target": [ancillas_names[i]], "control_sequence": [1, 1]}]
            gate_sequence += [{"name": "X", "control": [n_qubits - 1 - i], "target": [2 * n_qubits - 1 - i], "control_sequence": [1]}]
            gate_sequence += [{"name": "X", "control": [ancillas_names[i - 1], 2 * n_qubits - 1 - i], "target": [ancillas_names[i]],
                               "control_sequence": [1, 1]}]

        # the final layer
        gate_sequence += [
            {"name": "X", "control": [0], "target": [n_qubits], "control_sequence": [1]}]
        gate_sequence += [
            {"name": "X", "control": [ancillas_names[n_qubits-2]], "target": [n_qubits], "control_sequence": [1]}]

        # the backward pass
        for i in range(n_qubits - 2, 0, -1):
            gate_sequence += [{"name": "X", "control": [ancillas_names[i - 1], 2 * n_qubits - 1 - i], "target": [ancillas_names[i]],
                               "control_sequence": [1, 1]}]
            gate_sequence += [
                {"name": "X", "control": [n_qubits - 1 - i, 2 * n_qubits - 1 - i], "target": [ancillas_names[i]], "control_sequence": [1, 0]}]
            gate_sequence += [
                {"name": "X", "control": [ancillas_names[i-1]], "target": [2 * n_qubits - 1 - i], "control_sequence": [1]}]

        # the last gate to return last ancillas to |0>
        gate_sequence += [{"name": "X", "control": [n_qubits - 1, 2 * n_qubits - 1], "target": [ancillas_names[0]], "control_sequence": [1, 0]}]

    elif target_register == 0:  # U_SUM|i>|j>=U_SUM|i+j modulo 2>|j>

        # addition of the least qubits
        gate_sequence += [{"name": "X", "control": [n_qubits - 1, 2 * n_qubits - 1], "target": [ancillas_names[0]], "control_sequence": [1, 1]}]
        gate_sequence += [{"name": "X", "control": [2 * n_qubits - 1], "target": [n_qubits - 1], "control_sequence": [1]}]

        # the forward pass
        for i in range(1, n_qubits - 1):
            gate_sequence += [
                {"name": "X", "control": [n_qubits - 1 - i, 2 * n_qubits - 1 - i], "target": [ancillas_names[i]], "control_sequence": [1, 1]}]
            gate_sequence += [{"name": "X", "control": [2 * n_qubits - 1 - i], "target": [n_qubits - 1 - i], "control_sequence": [1]}]
            gate_sequence += [{"name": "X", "control": [ancillas_names[i - 1], n_qubits - 1 - i], "target": [ancillas_names[i]],
                               "control_sequence": [1, 1]}]

        # the final layer
        gate_sequence += [
            {"name": "X", "control": [n_qubits], "target": [0], "control_sequence": [1]}]
        gate_sequence += [
            {"name": "X", "control": [ancillas_names[n_qubits-2]], "target": [0], "control_sequence": [1]}]

        # the backward pass
        for i in range(n_qubits - 2, 0, -1):
            gate_sequence += [{"name": "X", "control": [ancillas_names[i - 1], n_qubits - 1 - i], "target": [ancillas_names[i]],
                               "control_sequence": [1, 1]}]
            gate_sequence += [
                {"name": "X", "control": [n_qubits - 1 - i, 2 * n_qubits - 1 - i], "target": [ancillas_names[i]], "control_sequence": [0, 1]}]
            gate_sequence += [{"name": "X", "control": [ancillas_names[i - 1]], "target": [n_qubits - 1 - i], "control_sequence": [1]}]

        # the last gate to return last ancillas to |0>
        gate_sequence += [{"name": "X", "control": [n_qubits - 1, 2 * n_qubits - 1], "target": [ancillas_names[0]], "control_sequence": [0, 1]}]


    for i in range(n_qubits-1):
        gate_sequence+=[{"name":"kill_ancilla","parameter":ancillas_names[i]}]
    return gate_sequence


def U_lA(n_qubits, l, sparse_indices, order):
    """
    This is U^{(l)}_A from the paper https://arxiv.org/pdf/2405.12855 appendix A, it transforms the sparse index
    into the first row index of the Banded-Sparse matrix (row i is the i-th time permuted first row) |0>|r> --> |i>

    parameters:
    l is the sparse register
    n_qubits -- number of main register qubits
    if order ==0:|0>|r> --> |i> the sparse register is the first one
    if order==1:|r>|0> --> |i> the sparse register is the second one
    sparse_indexes -- array of sparse indexes of the firs row (r_s0 from the paper) IN THE DECIMAL FORMAT

    Totally the circuit use:
    1)n_qubits -- number of main register qubits
    2)1 -- pure ancilla

    len(sparse_indexes)<=2^l
    """
    if not all(x < y for x, y in zip(sparse_indices, sparse_indices[1:])):
        raise ValueError("The sequence sparse_indexes is not a strictly increasing sequence.")

    ancilla_name = create_name_for_ancilla()
    gate_sequence = [{"name": "create_ancilla", "parameter": ancilla_name}]

    for i in range(len(sparse_indices) - 1, -1, -1):
        if i != sparse_indices[i]:
            gate_sequence += Urs0(n_qubits, l, i, sparse_indices[i], ancilla_name)[1:-1]

    gate_sequence += [{"name": "kill_ancilla", "parameter": ancilla_name}]

    if order == 1:
        gate_sequence = [{'name': 'zoom_in', 'block_gate_sequence': gate_sequence,
                          'target': list(range(l, n_qubits)) + list(range(l)) + list(range(n_qubits, 2 * n_qubits))}]
        if n_qubits - l < l:
            for a in np.arange(n_qubits - l):
                gate_sequence += [{'name': 'swap', 'target': [a, l + a]}]
            for a in np.arange(l, n_qubits):
                for b in np.arange(2*l-n_qubits):
                    gate_sequence += [{'name': 'swap', 'target': [a - b - 1, a - b]}]
        elif n_qubits - l == l:
            for a in np.arange(l):
                gate_sequence += [{'name': 'swap', 'target': [a, a + l]}]
        elif n_qubits - l > l:
            for a in np.arange(l):
                gate_sequence += [{'name': 'swap', 'target': [a, n_qubits - l + a]}]
            for a in np.arange(l, n_qubits - l):
                for b in np.arange(l):
                    gate_sequence += [{'name': 'swap', 'target': [a - b - 1, a - b]}]
    return gate_sequence


def Urs0(n_qubits, l, s, column_index, ancilla_name):
    """
    This function construct the auxiliary unitary U_s^rs0, it transforms the sparse index `s` into the
    first row index `sparse_index` of the Banded-Sparse matrix (row i is the i-th time permuted first row)
    |0>|s> --> |sparse_index>. See https://arxiv.org/pdf/2405.12855 appendix A Eq.(5) for more details.

    parameters:
    l is the sparse register
    n_qubits -- number of main register qubits
    s -- the sparse index (IN THE DECIMAL FORMAT)
    sparse_index -- the column index of the s-th non-zero element in the first row (IN THE DECIMAL FORMAT)
    ancilla_name -- the name of the ancilla qubit 'a', see figure 8 in https://arxiv.org/pdf/2405.12855
    """
    gate_sequence = [{"name": "create_ancilla", "parameter": ancilla_name}]

    # which state need to be controlled for the first multi-controlled x SELECT|0>|s>
    first_control_sequence = [0 for i in range(n_qubits - l)] + list(map(int, binary_form_of_integer(s, l)))

    # SELECT|rs0>
    last_control_sequence = list(map(int, binary_form_of_integer(column_index, n_qubits)))

    # X/I sequence where to apply X according to the second layer of the figure 8 in article https://arxiv.org/pdf/2405.12855
    # compare first_control_sequence and last_control_sequence if a[i]!=b[i] need X gate on that i-th qubit (XOR logical)
    X_target_sequence = np.bitwise_xor(first_control_sequence, last_control_sequence).tolist()

    # first layer
    gate_sequence += [{"name": "X", "control": [i for i in range(n_qubits)], "target": [ancilla_name],
                      "control_sequence": first_control_sequence}]

    # layer of X/I see figure 8
    for i in range(n_qubits):
        if X_target_sequence[i] == 1:
            gate_sequence += [{"name": "X", "control": [ancilla_name], "target": [i], "control_sequence": [1]}]

    # last layer
    gate_sequence += [{"name": "X", "control": [i for i in range(n_qubits)], "target": [ancilla_name],
                       "control_sequence": last_control_sequence}]

    gate_sequence += [{"name": "kill_ancilla", "parameter": ancilla_name}]
    return gate_sequence
