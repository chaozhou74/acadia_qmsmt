import functools
import numpy as np

@functools.cache
def _digits_in_base(base: int, width: int) -> np.ndarray:
    """
    Generates all integers 0 to base**width - 1 expressed as base-`base` digits.
    
    **Example**
        >>> _digits_in_base(2, 3)
        array([[0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]])
    """
    n = base ** width
    digits = np.unravel_index(np.arange(n), (base,) * width)
    return np.array(digits).T


@functools.cache
def make_pauli_labels(n_qubits):
    """
    Generate all 4**n_qubits pauli strings for n_qubits in lexicographic order.
    """
    pauli_indicies = _digits_in_base(4, n_qubits)
    labels = [''.join(np.array(["I", 'X', 'Y', 'Z'])[row]) for row in pauli_indicies]
    return labels


def parse_symmetrized_data(*shots_arr):
    '''
    Symmetrize the expectation values of Pauli ops by accounting for measurement biases between antiparallel
    Bloch sphere axes. For n qubits, this function searches for 2^n measurements done per expectation value (two 
    antiparallel axes per single-qubit tomography unitary), and appropriately signs the outcome shots.
    '''
    n_qubits = len(shots_arr)

    # shots_arr below has shape (n_qubits, num_iterations, 3^n_qubits OR 6^n_qubits)
    if type(shots_arr[0]) == np.ma.masked_array:
        shots_arr = np.ma.masked_array(shots_arr)
    else:
        shots_arr = np.array(shots_arr)
    
    # if trailing dim is 3^n_qubits, symmetrized data was not taken so we just return the shots_arr
    if np.shape(shots_arr)[-1] == 3**n_qubits:
        return shots_arr
    if np.shape(shots_arr)[-1] != 6**n_qubits:
        raise ValueError('Size of shots array matches neither symmetrized nor unsymmetrized constraint.')
    
    # reshape overcomplete data to correspond to 2^n_qubits sets of msmts per unique expectation value
    shots_arr_2d = shots_arr.reshape(n_qubits, -1, 3**n_qubits, 2**n_qubits)

    # generate signs for each of the (2^n_qubits) msmts based on whether they should return the same outcome
    # as the principal msmt 
    bin_nums = [f"{i:0{n_qubits}b}" for i in range(2**n_qubits)]
    sign_vector = np.array([[(-1)**int(b) for b in bin_num] for bin_num in bin_nums]).T.reshape(n_qubits, 1, 1, -1)

    # Apply the new signs to the original shots_arr and package these adjusted outcomes into the 'num_iterations'
    # dimension, meaning the symmetrized msmts get treated as additional msmts taken on the principal operator
    shots_arr_signed = shots_arr_2d * sign_vector
    shots_arr_signed = np.moveaxis(shots_arr_signed, -1, -2).reshape(n_qubits, -1, 3**n_qubits)

    return shots_arr_signed 


def xyz_to_full_tomo(*shots_array):
    """
    Reconstruct ⟨P⟩ for all P ∈ {I,X,Y,Z}^⊗N from ±1 outcomes measured in all {X,Y,Z}^⊗N settings.

    :param shots_array: array of measured shots for each qubit, each with shape (n_iter, 3**n_qubits)
        The measurements are assumed to be done in the order of from the most significant qubit 
        to the least (consistent with _digits_in_base).

    """
    meas_xyz = parse_symmetrized_data(*shots_array)
    N, R, B = meas_xyz.shape

    # Build compatibility matrix via Kronecker product
    M1 = np.array([
        [1, 1, 1],  # I
        [1, 0, 0],  # X
        [0, 1, 0],  # Y
        [0, 0, 1],  # Z
    ], dtype=bool)
    M = M1
    for _ in range(N - 1):
        M = np.kron(M, M1)   # (4^k, 3^k) -> (4^(k+1), 3^(k+1))
    # M is now (4**N, 3**N) boolean

    # Base-4 digits for Pauli rows; active qubits are those with digit != 0 (i.e., not I)
    pauli_digits = _digits_in_base(4, N)      # (4**N, N) with {I=0,X=1,Y=2,Z=3}
    active_mask = (pauli_digits != 0)         # (P, N) booleans
    
    labels = make_pauli_labels(N)
    exps = {}
    exps[labels[0]] = np.array([1.0]) # by defination ⟨I^⊗N⟩ = 1

    # For each Pauli p, compatible basis indices: b_idx = np.flatnonzero(M[p])
    for p in range(1, 4**N):
        act = active_mask[p]                   # (N,)
        b_idx = np.flatnonzero(M[p])          # (n_compat,)
        selected = meas_xyz[act][:, :, b_idx]  # (n_active, R, n_compat)

        if isinstance(selected, np.ma.MaskedArray):
            union_mask = np.any(np.ma.getmaskarray(selected), axis=0, keepdims=True)
            selected = np.ma.masked_array(selected.data, mask=np.broadcast_to(union_mask, selected.shape))

        prod_rb = selected.prod(axis=0)
        exps[labels[p]] = prod_rb

    return exps



def expvals_to_rho(expval_dict):
    '''
    Reconstruct the n-qubit density matrix from (2^2n)-1 measured Pauli operator expectation values.

    :param expval_dict: A dictionary with keys being the n-qubit Pauli operator expressed as a n-char string
                        and values being the corresponding shot value averaged over all iterations.

    :return: The density matrix expressed as a numpy array of shape (2^n, 2^n).
    '''
    import qutip
    pauli_dict = {"I": qutip.qeye(2),
                  "X": qutip.sigmax(),
                  "Y": qutip.sigmay(),
                  "Z": qutip.sigmaz()}
    
    # generate all measurable Pauli strings, exclude pure identity
    n_qubits = len(list(expval_dict.keys())[0])
    pauli_labels = make_pauli_labels(n_qubits)[1:]

    expvals = np.array([expval_dict[k] for k in pauli_labels])

    def pauli_str_to_op(pauli_str):
        '''
        Returns the (tensored) Qobj corresponding to a Pauli operator string.
        '''
        return qutip.tensor([pauli_dict[p] for p in pauli_str])
    
    num_expvals = len(expvals)
    rho_dim = int(np.round(np.sqrt(num_expvals + 1)))
    nqb_pauli_ops = [pauli_str_to_op(pauli_str) for pauli_str in pauli_labels]
    
    # generate rho by adding terms like (expectation of Pauli_op) * Pauli_op
    rho = np.zeros((rho_dim, rho_dim), dtype='complex128')
    for i in range(num_expvals): 
        rho += expvals[i] * nqb_pauli_ops[i].full()
    rho *= 0.5**n_qubits # normalize

    # add pure identity term to rho based on trace-preserving constraint
    exp_In = (1 - np.trace(rho))/rho_dim
    rho += exp_In * np.identity(rho_dim)
    
    return rho