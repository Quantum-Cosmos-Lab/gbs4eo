import numpy as np

def pyramid_statevector(n_layers):
    """
    Full 2^N state vector after a size-n BS pyramid on a single excitation.
    """
    N = 2 * n_layers
    p0 = n_layers - 1  # initial excitation qubit

    # --- single-particle amplitudes (O(n²)) ---
    c = np.zeros(N, dtype=complex)
    c[p0] = 1.0

    BS = np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2)

    for L in range(1, n_layers + 1):
        for off in range(-(L - 1), L, 2):
            i = p0 + off
            c[i], c[i + 1] = BS @ c[i:i + 2]

    # --- embed into full Hilbert space ---
    sv = np.zeros(2**N, dtype=complex)
    for k in range(N):
        sv[1 << (N - 1 - k)] = c[k]   # |0…1_k…0⟩

    return sv

if __name__ == '__main__':
    n_layers = 2
    sv = pyramid_statevector(n_layers)
    print(f'State vector for pyramyd with {n_layers} layers is {sv}')