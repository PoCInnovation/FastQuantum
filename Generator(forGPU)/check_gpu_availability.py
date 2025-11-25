"""
GPU Availability Diagnostic for Qiskit Quantum Simulation
Checks if your system can use GPU acceleration with Qiskit Aer

Requirements for GPU:
    - NVIDIA GPU with CUDA support
    - CUDA Toolkit installed
    - qiskit-aer-gpu OR qiskit-aer with GPU support
"""

import sys
import platform

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    print("\n" + "="*70)
    print("🔍 CHECKING NVIDIA GPU")
    print("="*70)

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            print("\nGPU Information:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi command failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - No NVIDIA GPU or drivers not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def check_cuda():
    """Check if CUDA is available"""
    print("\n" + "="*70)
    print("🔍 CHECKING CUDA")
    print("="*70)

    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Toolkit installed!")
            print(result.stdout)
            return True
        else:
            print("❌ CUDA not found")
            return False
    except FileNotFoundError:
        print("❌ nvcc not found - CUDA Toolkit not installed")
        print("\nInstall CUDA from: https://developer.nvidia.com/cuda-downloads")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_qiskit_aer_gpu():
    """Check if Qiskit Aer with GPU support is available"""
    print("\n" + "="*70)
    print("🔍 CHECKING QISKIT AER GPU SUPPORT")
    print("="*70)

    try:
        from qiskit_aer import AerSimulator

        # Try to create GPU simulator
        try:
            simulator = AerSimulator(method='statevector', device='GPU')
            print("✅ Qiskit Aer GPU support available!")
            print(f"   Simulator: {simulator}")

            # Get available devices
            available_devices = AerSimulator().available_devices()
            print(f"   Available devices: {available_devices}")

            return True
        except Exception as e:
            print(f"❌ Qiskit Aer installed but GPU not available: {e}")
            print("\nYou may need to install qiskit-aer-gpu:")
            print("   pip uninstall qiskit-aer")
            print("   pip install qiskit-aer-gpu")
            return False

    except ImportError:
        print("❌ Qiskit Aer not installed")
        print("\nInstall with:")
        print("   pip install qiskit-aer")
        print("   OR for GPU: pip install qiskit-aer-gpu")
        return False


def check_multiprocessing():
    """Check CPU multiprocessing capabilities"""
    print("\n" + "="*70)
    print("🔍 CHECKING CPU MULTIPROCESSING")
    print("="*70)

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"✅ CPU cores available: {cpu_count}")
    print(f"   Recommended parallel workers: {max(1, cpu_count - 1)}")
    return cpu_count


def test_simple_gpu_simulation():
    """Run a simple test to verify GPU actually works"""
    print("\n" + "="*70)
    print("🧪 TESTING GPU SIMULATION")
    print("="*70)

    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        import time

        # Create a small quantum circuit
        n_qubits = 10
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)

        # Test CPU
        print("\n⏱️  Testing CPU simulation...")
        simulator_cpu = AerSimulator(method='statevector', device='CPU')
        start = time.time()
        result_cpu = simulator_cpu.run(qc, shots=1).result()
        cpu_time = time.time() - start
        print(f"   CPU time: {cpu_time:.4f}s")

        # Test GPU
        try:
            print("\n⏱️  Testing GPU simulation...")
            simulator_gpu = AerSimulator(method='statevector', device='GPU')
            start = time.time()
            result_gpu = simulator_gpu.run(qc, shots=1).result()
            gpu_time = time.time() - start
            print(f"   GPU time: {gpu_time:.4f}s")

            speedup = cpu_time / gpu_time
            print(f"\n🚀 GPU Speedup: {speedup:.2f}x")

            if speedup > 1.5:
                print("   ✅ GPU is faster - ready to use!")
                return True
            else:
                print("   ⚠️  GPU not significantly faster (overhead for small circuits)")
                print("   💡 GPU will be faster for larger circuits (15+ qubits)")
                return True

        except Exception as e:
            print(f"   ❌ GPU test failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False


def main():
    """Run all diagnostics"""
    print("="*70)
    print("🚀 QISKIT GPU ACCELERATION DIAGNOSTIC")
    print("="*70)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print("="*70)

    results = {
        'gpu': check_nvidia_gpu(),
        'cuda': check_cuda(),
        'qiskit_gpu': check_qiskit_aer_gpu(),
        'cpu_cores': check_multiprocessing()
    }

    # Run simulation test if Qiskit GPU is available
    if results['qiskit_gpu']:
        results['gpu_test'] = test_simple_gpu_simulation()
    else:
        results['gpu_test'] = False

    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)

    if results['gpu'] and results['cuda'] and results['qiskit_gpu'] and results['gpu_test']:
        print("✅ GPU ACCELERATION FULLY READY!")
        print("\n🚀 You can use GPU-accelerated quantum dataset generation!")
        print("\nRecommended configuration:")
        print("   - Use GPU for quantum simulations")
        print(f"   - Use {max(1, results['cpu_cores'] - 1)} CPU workers for parallelization")
        print("   - Expected speedup: 10-100x for large graphs")

    elif results['gpu'] and not results['cuda']:
        print("⚠️  GPU detected but CUDA not installed")
        print("\nInstall CUDA Toolkit:")
        print("   https://developer.nvidia.com/cuda-downloads")

    elif results['gpu'] and results['cuda'] and not results['qiskit_gpu']:
        print("⚠️  GPU and CUDA ready, but Qiskit Aer GPU not available")
        print("\nInstall Qiskit Aer GPU:")
        print("   pip uninstall qiskit-aer")
        print("   pip install qiskit-aer-gpu")

    else:
        print("❌ GPU acceleration not available")
        print("\n💡 Fallback to CPU multiprocessing:")
        print(f"   - Use {max(1, results['cpu_cores'] - 1)} parallel workers")
        print("   - Expected speedup: 4-8x")

    print("\n" + "="*70)
    print(" ! Need cuquantum ! please install it before")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    main()
