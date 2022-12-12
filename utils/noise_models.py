import cirq


class DecoherenceNoiseModel(cirq.NoiseModel):
    def __init__(
            self,
            one_qubit_depol_error_rate=0.001,
            one_qubit_damping_error_rate=0.001,
            two_qubit_depol_error_rate=0.001):

        self.p = one_qubit_depol_error_rate
        self.p2 = two_qubit_depol_error_rate
        self.gamma = one_qubit_damping_error_rate

    def noisy_operation(self, op):
        n_qubits = len(op.qubits)

        depol_error_rate = self.p if n_qubits == 1 else self.p2
        depolarize_channel = cirq.depolarize(depol_error_rate, n_qubits=n_qubits)
        damping_channel = cirq.amplitude_damp(self.gamma)

        if n_qubits == 1:
            return [
                op, depolarize_channel.on(*op.qubits),
                damping_channel.on(*op.qubits)]
        elif n_qubits == 2:
            return [
                op, depolarize_channel.on(*op.qubits),
                damping_channel.on(op.qubits[0]),
                damping_channel.on(op.qubits[1])]
        else:
            return op


def with_decoherence_and_readout_noise(
        circuit,
        qubits,
        one_qubit_depol_p=0.,
        two_qubit_depol_p=0.,
        damping_gamma=0.,
        bitflip_readout=0.):
    """Adds decoherence noise and readout error to a PQC"""

    if (one_qubit_depol_p > 1e-5) or (two_qubit_depol_p > 1e-5) or (damping_gamma > 1e-5):
        circuit = circuit.with_noise(
            DecoherenceNoiseModel(
                one_qubit_depol_error_rate=one_qubit_depol_p,
                two_qubit_depol_error_rate=two_qubit_depol_p,
                one_qubit_damping_error_rate=damping_gamma
            ))

    if bitflip_readout > 1e-5:
        readout_error = cirq.BitFlipChannel(bitflip_readout)
        circuit += readout_error.on_each(qubits)

    return circuit
