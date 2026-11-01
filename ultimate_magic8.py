"""ultimate_magic8.py -- exact tools for "8次魔方陣の究極指数" (ultimate index of order-8 magic squares).

Everything is exact integer arithmetic.  The centred matrix N = A - (65/2)E has half-integer entries,
so the code works with B = 2N = 2A - 65E (an integer matrix).  N^k = 0  <=>  B^k = 0  <=>  A^k is a constant matrix.

Besides the order-8 tools it contains small helpers for order 4 (used only for Remark 8 of the paper, the F_109 phenomenon).

Conventions (same as the paper):
  * E: all-ones 8x8 matrix, J: reversal matrix, S: cyclic shift with S[i, i+1] = 1 (indices mod 8).
  * (S^r X S^{-c})[i][j] = X[i+r][j+c]  (indices mod 8).
  * X^(sigma)[i][j] = X[sigma(i)][sigma(j)]  (the same permutation on rows and columns; a similarity P X P^T).
  * 'nu' (究極指数, ultimate index) = smallest k such that A^k is a constant matrix; None means infinity.
"""
import itertools
from fractions import Fraction

import numpy as np

n = 8
LINE_SUM = 260          # magic constant L
COMPL = 65              # n^2 + 1
E = np.ones((n, n), dtype=np.int64)


def _m(rows):
    return np.array(rows, dtype=np.int64)


# ---------------------------------------------------------------- the examples of the paper
TYPE_I = _m([[1, 2, 59, 60, 61, 62, 7, 8], [16, 15, 54, 53, 52, 51, 10, 9], [17, 18, 43, 44, 45, 46, 23, 24],
             [32, 31, 38, 37, 36, 35, 26, 25], [40, 39, 30, 29, 28, 27, 34, 33], [41, 42, 19, 20, 21, 22, 47, 48],
             [56, 55, 14, 13, 12, 11, 50, 49], [57, 58, 3, 4, 5, 6, 63, 64]])          # symmetric (associative)
TYPE_II = _m([[1, 16, 17, 32, 57, 56, 41, 40], [2, 15, 18, 31, 58, 55, 42, 39], [59, 54, 43, 38, 3, 14, 19, 30],
              [60, 53, 44, 37, 4, 13, 20, 29], [8, 9, 24, 25, 64, 49, 48, 33], [7, 10, 23, 26, 63, 50, 47, 34],
              [62, 51, 46, 35, 6, 11, 22, 27], [61, 52, 45, 36, 5, 12, 21, 28]])        # pandiagonal ("M" in the paper)
TYPE_III = _m([[14, 3, 62, 51, 46, 35, 30, 19], [52, 61, 4, 13, 20, 29, 36, 45], [11, 6, 59, 54, 43, 38, 27, 22],
               [53, 60, 5, 12, 21, 28, 37, 44], [55, 58, 7, 10, 23, 26, 39, 42], [9, 8, 57, 56, 41, 40, 25, 24],
               [50, 63, 2, 15, 18, 31, 34, 47], [16, 1, 64, 49, 48, 33, 32, 17]])       # Franklin-type (Loly-Cameron-Rogers, corrected)

# 2H, where H is the half-integer 8x8 matrix in the remark after Theorem 11 of Hill-Lettington-Schmidt
# (arXiv:1605.08629; transcribed by the author of the paper).  H + 65/2 is a natural symmetric magic square.
HLS_TWICE = _m([[63, -61, 53, -55, -57, 59, -51, 49], [-47, 45, -37, 39, 41, -43, 35, -33],
                [-31, 29, -21, 23, 25, -27, 19, -17], [15, -13, 5, -7, -9, 11, -3, 1],
                [-1, 3, -11, 9, 7, -5, 13, -15], [17, -19, 27, -25, -23, 21, -29, 31],
                [33, -35, 43, -41, -39, 37, -45, 47], [-49, 51, -59, 57, 55, -53, 61, -63]])
HLS_SQUARE = (HLS_TWICE + COMPL) // 2

# 8th-order examples of Loly-Cameron-Trump-Schindel (Linear Algebra Appl. 430 (2009)), transcribed:
WOODRUFF = _m([[1, 32, 34, 63, 37, 60, 6, 27], [48, 49, 15, 18, 12, 21, 43, 54], [19, 14, 52, 45, 55, 42, 24, 9],
               [62, 35, 29, 4, 26, 7, 57, 40], [25, 8, 58, 39, 61, 36, 30, 3], [56, 41, 23, 10, 20, 13, 51, 46],
               [11, 22, 44, 53, 47, 50, 16, 17], [38, 59, 5, 28, 2, 31, 33, 64]])
MAGIC8 = _m([[64, 9, 17, 40, 32, 41, 49, 8], [2, 55, 47, 26, 34, 23, 15, 58], [3, 54, 46, 27, 35, 22, 14, 59],
             [61, 12, 20, 37, 29, 44, 52, 5], [60, 13, 21, 36, 28, 45, 53, 4], [6, 51, 43, 30, 38, 19, 11, 62],
             [7, 50, 42, 31, 39, 18, 10, 63], [57, 16, 24, 33, 25, 48, 56, 1]])         # a phase of MATLAB magic(8)

# vectors of the rank-2 factorisation 2N = e g^T + p f^T of the centred matrix of TYPE_II
E_VEC = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int64)
P_VEC = np.array([-15, -13, 5, 7, -1, -3, 11, 9], dtype=np.int64)
G_VEC = np.array([-48, -48, -16, -16, 64, 32, 32, 0], dtype=np.int64)
F_VEC = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)

# ---------------------------------------------------------------- matrices S, J, Gamma
S = np.zeros((n, n), dtype=np.int64)
for _i in range(n):
    S[_i, (_i + 1) % n] = 1
J = np.fliplr(np.eye(n, dtype=np.int64))


def spow(k):
    return np.linalg.matrix_power(S, k % n)


GAMMA = [("S^%d" % d, spow(d)) for d in range(n)] + [("JS^%d" % e, J @ spow(e)) for e in range(n)]   # 16 matrices


# ---------------------------------------------------------------- predicates
def two_n(X):
    """B = 2N = 2X - 65E (integer matrix)."""
    return 2 * np.asarray(X, dtype=np.int64) - COMPL * E


def is_natural(X):
    return sorted(np.asarray(X).ravel().tolist()) == list(range(1, n * n + 1))


def is_semimagic(X):
    X = np.asarray(X)
    return bool((X.sum(0) == LINE_SUM).all() and (X.sum(1) == LINE_SUM).all())


def is_magic(X):
    X = np.asarray(X)
    return is_semimagic(X) and np.trace(X) == LINE_SUM and np.trace(np.fliplr(X)) == LINE_SUM


def is_associative(X):
    X = np.asarray(X)
    return bool((X + X[::-1, ::-1] == COMPL).all())


# flat indices of the 16 broken diagonals (8 down, 8 anti)
DIAG_IDX = np.array([[i * n + (i + k) % n for i in range(n)] for k in range(n)] +
                    [[i * n + (k - i) % n for i in range(n)] for k in range(n)], dtype=np.int64)


def is_pandiagonal(X):
    """pandiagonal magic square: rows, columns and all 16 broken diagonals sum to 260."""
    X = np.asarray(X)
    flat = X.ravel()
    return is_semimagic(X) and bool((flat[DIAG_IDX].sum(1) == LINE_SUM).all())


def has_antipodal_complement(X):
    """(C): X[a+4][b+4] = 65 - X[a][b]  (indices mod 8)."""
    X = np.asarray(X)
    return bool((np.roll(np.roll(X, -4, 0), -4, 1) == COMPL - X).all())


# ---------------------------------------------------------------- exact linear algebra
def _is_zero(P):
    return all(v == 0 for v in P.flat)


def ultimate_index(X, kmax=n):
    """nu(X): least k with N^k = 0 (<=> X^k constant).  None = infinity.  X must be semimagic with line sum 260."""
    B = np.array(two_n(X), dtype=object)
    P = B.copy()
    for k in range(1, kmax + 1):
        if _is_zero(P):
            return k
        P = P.dot(B)
    return None


def power_is_constant(X, k):
    P = np.array(X, dtype=object)
    Q = P.copy()
    for _ in range(k - 1):
        Q = Q.dot(P)
    return len(set(int(v) for v in Q.flat)) == 1, int(Q.flat[0])


def rank_exact(M):
    A = [[Fraction(int(x)) for x in row] for row in np.asarray(M, dtype=object).tolist()]
    rows, cols = len(A), len(A[0])
    rank = 0
    for c in range(cols):
        piv = next((r for r in range(rank, rows) if A[r][c] != 0), None)
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        for r in range(rank + 1, rows):
            if A[r][c] != 0:
                f = A[r][c] / A[rank][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[rank])]
        rank += 1
    return rank


def charpoly(X):
    """coefficients [1, c1, ..., cn] of det(xI - X) (Faddeev-LeVerrier, exact integers)."""
    A = np.array(X, dtype=object)
    m = A.shape[0]
    I = np.array(np.eye(m, dtype=np.int64), dtype=object)
    Mk = np.zeros((m, m), dtype=object)
    coeffs = [1]
    for k in range(1, m + 1):
        Mk = A.dot(Mk) + coeffs[-1] * I
        t = sum(A.dot(Mk)[i, i] for i in range(m))
        if t % k != 0:
            raise ArithmeticError("Faddeev-LeVerrier trace not divisible by k: input is not an integer matrix")
        coeffs.append(-(t // k))
    return [int(c) for c in coeffs]


def jordan_type_of_N(X):
    """Jordan type (partition of 8) of the nilpotent 8x8 matrix N = (X - 65/2 E); None if N is not nilpotent."""
    B = np.array(two_n(X), dtype=object)
    ranks = [n]
    P = np.array(np.eye(n, dtype=np.int64), dtype=object)
    for _ in range(n + 1):
        P = P.dot(B)
        ranks.append(rank_exact(P))
        if ranks[-1] == 0:
            break
    if ranks[-1] != 0:
        return None
    d = [ranks[k - 1] - ranks[k] for k in range(1, len(ranks))] + [0]     # d_k = number of blocks of size >= k
    sizes = []
    for k in range(1, len(d)):
        sizes += [k] * (d[k - 1] - d[k])
    return sorted(sizes, reverse=True)


# ---------------------------------------------------------------- symmetries
def d4_images(X):
    """the eight D4 images, in the column order of Table 2 of the paper."""
    X = np.asarray(X)
    return {"X": X, "JX^T": J @ X.T, "JXJ": J @ X @ J, "X^TJ": X.T @ J,
            "X^T": X.T, "JX": J @ X, "JX^TJ": J @ X.T @ J, "XJ": X @ J}


TABLE2_COLUMNS = ["X", "JX^T", "JXJ", "X^TJ", "X^T", "JX", "JX^TJ", "XJ"]


def window(X, r, c):
    """M_{r,c} = S^r X S^{-c}: entry (i,j) is X[(i+r)%8][(j+c)%8]."""
    X = np.asarray(X)
    return X[np.ix_([(i + r) % n for i in range(n)], [(j + c) % n for j in range(n)])]


def permute(X, sigma):
    """X^(sigma)[i][j] = X[sigma(i)][sigma(j)]."""
    X = np.asarray(X)
    s = list(sigma)
    return X[np.ix_(s, s)]


def sigma_set():
    """Sigma = {sigma : sigma(7-i) = sigma(i)+4 mod 8}: mirror pairs {i,7-i} -> antipodal pairs {a,a+4}. |Sigma| = 4! * 2^4 = 384."""
    pairs = [(a, a + 4) for a in range(4)]
    out = []
    for assign in itertools.permutations(range(4)):
        for bits in itertools.product((0, 1), repeat=4):
            s = [0] * n
            for i in range(4):
                a, b = pairs[assign[i]]
                if bits[i]:
                    a, b = b, a
                s[i], s[n - 1 - i] = a, b
            out.append(tuple(s))
    return out


def inverse_perm(p):
    q = [0] * len(p)
    for i, x in enumerate(p):
        q[x] = i
    return tuple(q)


def rho_set():
    """Sigma' = {rho : rho(a+4) = 7 - rho(a)} = Sigma^{-1}."""
    return [inverse_perm(s) for s in sigma_set()]


SIGMA0 = (0, 1, 2, 3, 7, 6, 5, 4)       # (5 8)(6 7) in 1-based numbering = Nordgren's W2 = [[I4,0],[0,R4]]
PLANCK_W1 = (3, 2, 1, 0, 4, 5, 6, 7)    # Nordgren's W1 = [[R4,0],[0,I4]]
PLANCK_W2 = SIGMA0


def perm_matrix(sigma):
    P = np.zeros((n, n), dtype=np.int64)
    for i, s in enumerate(sigma):
        P[i, s] = 1
    return P


# ---------------------------------------------------------------- batch helpers (uint8 flat squares)
def perm_index(perms):
    p = np.asarray(perms, dtype=np.uint8)
    return (p[:, :, None] * 8 + p[:, None, :]).reshape(len(p), 64)


def batch_associative(Y):
    Z = np.asarray(Y).reshape(-1, 8, 8).astype(np.int16)
    return (Z + Z[:, ::-1, ::-1] == COMPL).all(axis=(1, 2))


def batch_pandiagonal(Y):
    Z = np.asarray(Y).reshape(-1, 64).astype(np.int16)
    R = Z.reshape(-1, 8, 8)
    ok = (R.sum(1) == LINE_SUM).all(1) & (R.sum(2) == LINE_SUM).all(1)
    ok &= (Z[:, DIAG_IDX].sum(2) == LINE_SUM).all(1)
    return ok


def unique_rows(Y):
    Y = np.ascontiguousarray(np.asarray(Y, dtype=np.uint8))
    v = Y.view(np.dtype((np.void, Y.shape[1]))).ravel()
    _, idx = np.unique(v, return_index=True)
    return Y[idx]


def batch_nu_le3(Y):
    """for a batch of flat semimagic squares: 2 if N^2=0, 3 if N^3=0 and N^2!=0, 0 otherwise (nu>3 or infinity)."""
    B = 2 * np.asarray(Y, dtype=np.int64).reshape(-1, 8, 8) - COMPL
    B2 = B @ B
    B3 = B2 @ B
    z2 = (B2 == 0).all(axis=(1, 2))
    z3 = (B3 == 0).all(axis=(1, 2))
    return np.where(z2, 2, np.where(z3, 3, 0))


TRANS_IDX = np.array([[((i + r) % n) * n + (j + c) % n for i in range(n) for j in range(n)]
                      for r in range(n) for c in range(n)], dtype=np.int64)          # row index = 8*r + c


# ---------------------------------------------------------------- order 4 (Remark 8 of the paper: rotation can create ultimacy)
# F_109 of Loly-Cameron-Trump-Schindel, Linear Algebra Appl. 430 (2009), equation (37) (a pandiagonal 4x4 square):
F109 = _m([[1, 8, 11, 14], [15, 10, 5, 4], [6, 3, 16, 9], [12, 13, 2, 7]])


def ultimate_index_n(X, kmax=None):
    """nu(X) for a semimagic n x n integer matrix X of any order n (line sum n(n^2+1)/2); None = infinity.
    Uses B = 2N = 2X - (n^2+1)E, an integer matrix: N^k = 0  <=>  B^k = 0  <=>  X^k is constant."""
    X = np.asarray(X, dtype=np.int64)
    m = X.shape[0]
    B = np.array(2 * X - (m * m + 1) * np.ones((m, m), dtype=np.int64), dtype=object)
    P = B.copy()
    for k in range(1, (kmax or m) + 1):
        if _is_zero(P):
            return k
        P = P.dot(B)
    return None


def pandiagonal_4x4():
    """ALL pandiagonal natural magic squares of order 4 (there are 384), found exhaustively: the 16 line conditions
    (4 rows, 4 columns, 4+4 broken diagonals; the two main diagonals are among the latter) are linear, so every solution
    is determined by the 4 free cells of the row-reduced system; all 16^4 values of those cells are tried and the results
    are filtered for integrality and for using 1..16 exactly once.  Returns an array of shape (384, 4, 4)."""
    m, L = 4, 34
    lines = []
    for i in range(m):
        lines.append([i * m + j for j in range(m)])
        lines.append([j * m + i for j in range(m)])
    for k in range(m):
        lines.append([i * m + (i + k) % m for i in range(m)])
        lines.append([i * m + (k - i) % m for i in range(m)])
    M = [[Fraction(int(c in ln)) for c in range(16)] + [Fraction(L)] for ln in lines]
    piv, r = [], 0
    for c in range(16):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        M[r] = [x / M[r][c] for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(16) if c not in piv]
    den = 1
    for i in range(r):
        for x in M[i]:
            den = int(np.lcm(den, x.denominator))
    vals = np.array(list(itertools.product(range(1, 17), repeat=len(free))), dtype=np.int64)
    X = np.zeros((len(vals), 16), dtype=np.int64)
    X[:, free] = vals
    for i, p in enumerate(piv):
        num = int(M[i][16] * den) * np.ones(len(vals), dtype=np.int64)
        for fi, f in enumerate(free):
            num -= int(M[i][f] * den) * vals[:, fi]
        X[:, p] = np.where(num % den == 0, num // den, -1)
    X = X[((X >= 1) & (X <= 16)).all(1)]
    X = X[(np.sort(X, 1) == np.arange(1, 17)).all(1)]
    return X.reshape(-1, 4, 4)
