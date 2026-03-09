import pennylane as qml
print("PennyLane imported.")
try:
    dev = qml.device("default.qubit", wires=2)
    print("Device created.")
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))
    print(f"Result: {circuit()}")
except Exception as e:
    print(f"Error: {e}")
