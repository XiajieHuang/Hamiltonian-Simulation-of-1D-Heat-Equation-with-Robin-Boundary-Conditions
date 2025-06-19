import numpy as np
from Sparse_Maplitude_Oracle_for_Momentum_Operator import O_Spm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit import transpile


def Ux(n, transpose):

    number_of_qubits = int(np.ceil(np.log2(n)))

    state_vector = np.zeros(2 ** number_of_qubits, dtype=np.complex128)
    for i in range(n):
        state_vector[i] = 1j * 2 ** (i/2) / np.sqrt(2 ** n - 1)
    if transpose:
        unitary = gram_schmidt(state_vector.conj())
        unitary = unitary.conj().T
    else:
        unitary = gram_schmidt(state_vector)

    #Use qiskit to decompose the unitary into one_qubits rotation and CNOT gate
    qc = QuantumCircuit(number_of_qubits)
    qc.unitary(Operator(unitary), list(range(number_of_qubits)), label="Unitary") # Convert the unitary matrix into a quantum circuit
    decomposed_circuit = transpile(qc, basis_gates=['u', 'cx'])# Decompose the circuit to get a more basic gate decomposition


    gate_sequence = []
    ## Extract the information of the decomposed gates
    # Iterate through the data of the decomposed quantum circuit
    for gate in decomposed_circuit.data:
        gate_name = gate[0].name  # Get the name of the gate
        params = gate[0].params  # Get the parameters of the gate
        qubits = [qc.find_bit(q).index for q in gate[1]]  # Get the qubits the gate acts on
        if gate_name == "cx":
            gate_sequence += [{'name' : 'x', 'target' : [number_of_qubits - 1 - qubits[1]],
                               'control' : [number_of_qubits - 1 - qubits[0]], 'control_sequence' : [1]}]
        else:
            gate_sequence += [{'name': 'u', 'parameter' : params, 'target': [number_of_qubits - 1 - qubits[0]],
                               'control': [], 'control_sequence': []}]
    global_phase = decomposed_circuit.global_phase
    gate_sequence += [{'name':'global_phase', 'parameter':global_phase, 'target':[0],'control':[],'control_sequence':[]}]

    return  number_of_qubits, gate_sequence

def Ux_general(n, a, b, transpose):
    """
    Auxiliary unitary for Ox whose domain is [a,b], b > a >= 0
    :param n:
    :param transpose:
    :return:
    """
    if b <= a :
        raise ValueError("b must be strictly greater than a!")
    factor = max(abs(a), abs(b))
    alpha = (a + b) / (2 * factor)
    beta = (b - a) / (2 * factor)


    number_of_qubits = int(np.ceil(np.log2(n + 1)))

    state_vector = np.zeros(2 ** number_of_qubits, dtype=np.complex128)
    for i in range(n):
        state_vector[i] = 1j * 2 ** (i/2) / np.sqrt(2 ** n - 1) * beta ** 0.5
    state_vector[n] = alpha ** 0.5
    if transpose:
        unitary = gram_schmidt(state_vector.conj())
        unitary = unitary.conj().T
    else:
        unitary = gram_schmidt(state_vector)

    #Use qiskit to decompose the unitary into one_qubits rotation and CNOT gate
    qc = QuantumCircuit(number_of_qubits)
    qc.unitary(Operator(unitary), list(range(number_of_qubits)), label="Unitary") # Convert the unitary matrix into a quantum circuit
    decomposed_circuit = transpile(qc, basis_gates=['u', 'cx'])# Decompose the circuit to get a more basic gate decomposition


    gate_sequence = []
    ## Extract the information of the decomposed gates
    # Iterate through the data of the decomposed quantum circuit
    for gate in decomposed_circuit.data:
        gate_name = gate[0].name  # Get the name of the gate
        params = gate[0].params  # Get the parameters of the gate
        qubits = [qc.find_bit(q).index for q in gate[1]]  # Get the qubits the gate acts on
        if gate_name == "cx":
            gate_sequence += [{'name' : 'x', 'target' : [number_of_qubits - 1 - qubits[1]],
                               'control' : [number_of_qubits - 1 - qubits[0]], 'control_sequence' : [1]}]
        else:
            gate_sequence += [{'name': 'u', 'parameter' : params, 'target': [number_of_qubits - 1 - qubits[0]],
                               'control': [], 'control_sequence': []}]
    global_phase = decomposed_circuit.global_phase
    gate_sequence += [{'name':'global_phase', 'parameter':global_phase, 'target':[0],'control':[],'control_sequence':[]}]

    return  number_of_qubits, gate_sequence


def gram_schmidt(a):
    """
    Create an n*n unitary matrix where the first column is replaced by the unit complex vector a,
    and the rest are the columns of the identity matrix.

    Parameters:
    a (numpy.ndarray): An n*1 unit complex vector.

    Returns:
    numpy.ndarray: An n*n unitary matrix.
    """
    if isinstance(a, list):
        a = np.array(a)

    # Create an n*n identity matrix with complex entries
    n = len(a) #n (int): The dimension of the matrix.
    U = np.eye(n, dtype=np.complex128)

    # Replace the first column with the vector a
    U[:, 0] = a.flatten()

    # Perform the Gram-Schmidt orthogonalization process
    for j in range(n):
        # Take the j-th column
        v = U[:, j]
        # Subtract its projection onto the already orthogonalized basis vectors
        for i in range(j):
            v -= np.dot(U[:, i].conj().T, v) * U[:, i]
        # Normalize the column
        U[:, j] = v / np.linalg.norm(v)

    return U