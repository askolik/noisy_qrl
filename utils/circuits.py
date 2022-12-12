import cirq
import sympy
import numpy as np


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


def generate_circuit(qubits, n_layers, noise_p=0.):
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
