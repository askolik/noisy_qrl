import cirq
import sympy
import numpy as np
import tensorflow_quantum as tfq


def graph_encoding_circuit(edges, qubits, n_layers, data_params):
    circuit = cirq.Circuit()
    circuit += cirq.H.on_each(qubits)

    for layer in range(n_layers):
        edge_weights = data_params[layer][-1]
        for edge_ix, edge in enumerate(edges):
            circuit.append(
                cirq.CNOT(qubits[edge[0]],
                          qubits[edge[1]]))

            circuit.append(cirq.rz(edge_weights[edge_ix])(qubits[edge[1]]))

            circuit.append(
                cirq.CNOT(qubits[edge[0]],
                          qubits[edge[1]]))

        for qubit_ix, qubit in enumerate(qubits):
            circuit += cirq.rx(data_params[layer][qubit_ix])(qubit)

    # print(circuit)
    # exit()

    return circuit


def one_qubit_rotation(qubit, symbols):
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]


def entangling_layer(qubits):
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


def generate_cp_pg_circuit(qubits, n_layers, noise_p=0.):
    """
    Prepares a data re-uploading circuit on `qubits` with `n_layers` layers.
    """
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3 * (n_layers + 1) * n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i, q in enumerate(qubits))

    # Add noise to the circuit
    if noise_p >= 1e-5:
        circuit = circuit.with_noise(cirq.depolarize(noise_p))

    return circuit, list(params.flat), list(inputs.flat)


def hwe_layer(qubits, symbols):
    circuit = cirq.Circuit()
    symbols = list(symbols)[::-1]

    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(symbols.pop())(qubit))
        circuit.append(cirq.rz(symbols.pop())(qubit))

    for i in range(len(qubits)):
        circuit.append(cirq.CZ(qubits[i], qubits[(i + 1) % len(qubits)]))

    return circuit


def generate_cp_q_circuit(n_qubits, n_layers, qubits, use_reuploading=True):
    theta_dim = 2 * n_qubits * n_layers
    params = sympy.symbols('theta(0:' + str(theta_dim) + ')')

    if use_reuploading:
        inputs = sympy.symbols(
            'x(0:' + str(n_qubits) + ')' + '(0:' + str(n_layers) + ')')
    else:
        inputs = sympy.symbols('x(0:' + str(n_qubits) + ')')

    circuit = cirq.Circuit()
    for l in range(n_layers):
        if use_reuploading:
            for i in range(n_qubits):
                circuit += cirq.rx(inputs[l + i * n_layers])(qubits[i])
        if not use_reuploading and l == 0:
            for i in range(n_qubits):
                circuit += cirq.rx(inputs[i])(qubits[i])

        circuit += hwe_layer(qubits, params[l * n_qubits * 2:(l + 1) * n_qubits * 2])

    return circuit, theta_dim, params, inputs


def empty_circuits(n):
    return tfq.convert_to_tensor([cirq.Circuit()]*n)