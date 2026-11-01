#!/usr/bin/env python3
"""verify_paper.py -- reproduces every numerical claim of the paper "8次魔方陣の究極指数".

Usage:
    python verify_paper.py            # everything (about 1.5 minutes)
    python verify_paper.py --quick    # skip the exhaustive orbit / Sigma / Sigma' enumeration (about 1 second)

Numbering.  The labels below ("Lemma 9", "Theorem 10", "Proposition 14", ...) are the numbers printed in the paper as
compiled from the TeX source, where definitions, lemmas, propositions, theorems, corollaries and remarks share ONE counter:
  Lemma 1 (powers), Definition 2, Lemma 3 (spectrum), Proposition 4 (rank 3 => nu in {2,3}), Remark 5 (Hill-Lettington-
  Schmidt), Theorem 6 (symmetric squares, D4), Corollary 7, Remark 8 (order 4, F_109), Lemma 9 (inner products),
  Theorem 10 (translates), Proposition 11 (Planck-type maps), Lemma 12 (products), Theorem 13 (products),
  Proposition 14 (counterexample).   Equations: (1) NE=EN=0, (2) JNJ=-N, (3) 2N = e g^T + p f^T, (4) N Z N = gamma/4 e f^T,
  (5) antipodal complement.   Tables: 1 examples, 2 D4 images, 3 gamma/eta, 4 counts, 5 translates.
If the TeX source is changed to number these per section, update the labels here accordingly.

All arithmetic is exact integer arithmetic (the centred matrix N has half-integer entries, so B = 2N is used).
Exit status 0 if and only if every check passes.
"""
import argparse
import collections
import itertools
import random
import sys
import time

import numpy as np

import ultimate_magic8 as um
from ultimate_magic8 import (E, J, S, GAMMA, E_VEC, P_VEC, G_VEC, F_VEC, SIGMA0, TABLE2_COLUMNS,
                             TYPE_I, TYPE_II, TYPE_III)

RESULTS = []


def check(label, condition, detail=""):
    RESULTS.append(bool(condition))
    print("  [%s] %s%s" % ("PASS" if condition else "FAIL", label, ("  " + str(detail)) if detail != "" else ""))
    return bool(condition)


def section(title):
    print("\n== %s" % title)


def outer(a, b):
    return np.outer(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the exhaustive enumerations (Tables 4 and 5 and the orbit statistics)")
    args = ap.parse_args()
    t0 = time.time()
    M = TYPE_II
    B = um.two_n(M)                                        # B = 2N
    IIb = um.window(M, 0, 2)                               # TYPE II' = M_{0,2}
    Ib = um.permute(IIb, SIGMA0)                           # TYPE I'  = (II')^{(sigma0)}

    # ------------------------------------------------------------------ Section 3: the examples
    section("Section 3 / Table 1: the examples")
    examples = [("I", TYPE_I, True, False, 3), ("II", TYPE_II, False, True, 3), ("II'", IIb, False, True, 2),
                ("I'", Ib, True, False, 2), ("III", TYPE_III, False, False, 3)]
    for name, X, assoc, pan, nu in examples:
        ok = (um.is_natural(X) and um.is_magic(X) and um.is_associative(X) == assoc and um.is_pandiagonal(X) == pan
              and um.rank_exact(X) == 3 and um.charpoly(X) == [1, -260] + [0] * 7 and um.ultimate_index(X) == nu)
        const_pow = um.power_is_constant(X, nu)
        jt = um.jordan_type_of_N(X)
        expected_jt = [3, 1, 1, 1, 1, 1] if nu == 3 else [2, 2, 1, 1, 1, 1]
        check("type %-3s natural magic, assoc=%s pandiag=%s, rank 3, charpoly x^7(x-260), nu=%d" % (name, assoc, pan, nu), ok)
        check("type %-3s A^%d constant = %d*E; Jordan type of N = %s" % (name, nu, const_pow[1], jt),
              const_pow[0] and jt == expected_jt and const_pow[1] == (2197000 if nu == 3 else 8450))
    check("Type I' is not a D4 image of Type I",
          all(not np.array_equal(Z, TYPE_I) for Z in um.d4_images(Ib).values()))
    for name, X in (("Woodruff", um.WOODRUFF), ("magic(8)", um.MAGIC8)):
        cp = um.charpoly(X)
        check("[LCTS] %s: natural magic, rank 3, nu = infinity, charpoly x^5 * (cubic with c3 != 0), i.e. exactly three nonzero eigenvalues" % name,
              um.is_natural(X) and um.is_magic(X) and um.rank_exact(X) == 3 and um.ultimate_index(X) is None
              and cp[-1] == 0 and cp[-2] == 0 and cp[-3] == 0 and cp[-4] == 0 and cp[-5] == 0 and cp[3] != 0, cp[:4])

    section("Lemma 1: (2A)^k = (2N)^k + 65^k 8^(k-1) E  (so A^k is constant iff (2N)^k = 0)")
    ok_l = True
    for X in (TYPE_I, TYPE_II, TYPE_III, IIb, Ib, um.WOODRUFF, um.MAGIC8):
        B0 = np.array(um.two_n(X), dtype=object)
        P2 = np.array(2 * X, dtype=object)
        Pk, Qk = P2.copy(), B0.copy()
        for k in range(1, 6):
            ok_l &= np.array_equal(Pk, Qk + 65 ** k * 8 ** (k - 1) * np.array(E, dtype=object))
            Pk, Qk = Pk.dot(P2), Qk.dot(B0)
    check("Lemma 1 for k = 1..5 on all examples (exact integer arithmetic)", ok_l)

    section("Remark 5 (Hill-Lettington-Schmidt): H + 65/2")
    H2 = um.HLS_TWICE
    Hs = um.HLS_SQUARE
    check("2H has zero row/column sums and H + 65/2 = HLS_SQUARE", (H2.sum(0) == 0).all() and (H2.sum(1) == 0).all()
          and np.array_equal(2 * Hs, H2 + 65))
    check("HLS_SQUARE: natural, symmetric (associative) magic square, nu = 3, A^3 = 2197000 E",
          um.is_natural(Hs) and um.is_magic(Hs) and um.is_associative(Hs) and um.ultimate_index(Hs) == 3
          and um.power_is_constant(Hs, 3) == (True, 2197000))
    check("HLS_SQUARE is not a D4 image of Type I", all(not np.array_equal(Z, TYPE_I) for Z in um.d4_images(Hs).values()))

    # ------------------------------------------------------------------ Remark 8 (order 4)
    section("Remark 8 (order 4): a 90-degree rotation can turn nu = infinity into nu = 3  [LCTS, F_109]")
    sq4 = um.pandiagonal_4x4()
    check("exhaustive: there are exactly 384 pandiagonal natural magic squares of order 4", len(sq4) == 384)
    phen = []
    for A in sq4:
        cpA = um.charpoly(A)
        three_nonzero = cpA[4] == 0 and cpA[3] != 0                      # char. poly x * (cubic with nonzero constant term)
        if three_nonzero and um.ultimate_index_n(A) is None:
            R = np.rot90(A)
            if um.ultimate_index_n(R) == 3 and um.charpoly(R) == [1, -34, 0, 0, 0]:
                phen.append(A)
    check("nu = infinity with exactly three nonzero eigenvalues, but the rotated square has nu = 3 and 34 as its only "
          "nonzero eigenvalue: occurs for exactly 64 of the 384 pandiagonal squares "
          "(F_109 of [LCTS] is among the 384 and is checked individually below)",
          len(phen) == 64, len(phen))
    check("the direction of the rotation is irrelevant (rotation by +90 and -90 differ by the similarity X -> JXJ)",
          all(um.ultimate_index_n(np.rot90(A, 1)) == um.ultimate_index_n(np.rot90(A, 3)) for A in phen))
    if um.F109 is not None:
        F = np.array(um.F109, dtype=np.int64)
        cpF = um.charpoly(F)
        check("F_109 (Loly et al. 2009, eq. (37)): one of the 384 enumerated squares, pandiagonal natural magic square, nu = infinity, three nonzero "
              "eigenvalues; its rotation has nu = 3 with charpoly x^3 (x - 34)",
              sorted(F.ravel().tolist()) == list(range(1, 17)) and any(np.array_equal(F, A) for A in sq4)
              and cpF[4] == 0 and cpF[3] != 0
              and um.ultimate_index_n(F) is None and um.ultimate_index_n(np.rot90(F)) == 3
              and um.charpoly(np.rot90(F)) == [1, -34, 0, 0, 0])
    else:
        print("  (F109 is not set in ultimate_magic8.py, so F_109 is not checked individually; the statement is verified for all 384 squares above)")

    # ------------------------------------------------------------------ Section 4
    section("Section 4: D4 images (Theorem 6, Corollary 7, Table 2)")
    for name, X in (("I", TYPE_I), ("I'", Ib), ("HLS", Hs)):
        N2 = um.two_n(X)
        check("%s: J N J = -N (symmetric squares), hence nu is constant on D4 images" % name,
              np.array_equal(J @ N2 @ J, -N2))
    table2 = {"I": TYPE_I, "II": TYPE_II, "II'": IIb, "I'": Ib, "III": TYPE_III}
    expected2 = {"I": [3] * 8, "II": [3] * 8, "II'": [2, 3, 2, 3, 2, 3, 2, 3], "I'": [2] * 8, "III": [3] * 8}
    print("  Table 2 (columns: %s)" % " ".join(TABLE2_COLUMNS))
    for name, X in table2.items():
        imgs = um.d4_images(X)
        row = [um.ultimate_index(imgs[c]) for c in TABLE2_COLUMNS]
        print("     %-4s %s" % (name, row))
        check("Table 2 row %s" % name, row == expected2[name])

    # ------------------------------------------------------------------ Section 5.1
    section("Section 5.1: the group Gamma, the families T and T^T, the rank-2 factorisation (equation (3))")
    check("Gamma has 16 distinct permutation matrices", len({Z.tobytes() for _, Z in GAMMA}) == 16)
    T_set = {(P @ M @ Q).astype(np.uint8).tobytes() for _, P in GAMMA for _, Q in GAMMA}
    TT_set = {(P @ M.T @ Q).astype(np.uint8).tobytes() for _, P in GAMMA for _, Q in GAMMA}
    check("|T| = 256 and |T^T| = 256 (all distinct)", len(T_set) == 256 and len(TT_set) == 256)
    check("T and T^T are disjoint", len(T_set & TT_set) == 0)
    trans = [um.window(M, r, c) for r in range(8) for c in range(8)]
    S1 = {Z.astype(np.uint8).tobytes(): Z for X in trans for Z in um.d4_images(X).values()}
    check("T union T^T = (64 translates) x (8 D4 images) = 512 distinct squares",
          len(S1) == 512 and set(S1) == (T_set | TT_set))
    check("every element of T is a pandiagonal natural magic square",
          all(um.is_pandiagonal(np.frombuffer(b, dtype=np.uint8).reshape(8, 8).astype(np.int64)) for b in T_set))
    check("2N = e g^T + p f^T, rank N = 2, row/column sums of N vanish",
          np.array_equal(B, outer(E_VEC, G_VEC) + outer(P_VEC, F_VEC)) and um.rank_exact(B) == 2
          and (B.sum(0) == 0).all() and (B.sum(1) == 0).all())
    ok_inner = all(G_VEC @ Z @ E_VEC == 0 and F_VEC @ Z @ E_VEC == 0 and F_VEC @ Z @ P_VEC == 0 for _, Z in GAMMA)
    check("Lemma 9: g^T Z e = f^T Z e = f^T Z p = 0 for all 16 Z in Gamma", ok_inner)
    gam = [int(G_VEC @ um.spow(d) @ P_VEC) for d in range(8)]
    eta = [int(G_VEC @ J @ um.spow(d) @ P_VEC) for d in range(8)]
    print("     gamma_delta =", gam)
    print("     eta_epsilon =", eta)
    check("Table 3 (gamma, eta)", gam == [1344, 736, 0, -736, -1344, -736, 0, 736]
          and eta == [-704, 480, 1024, 1056, 704, -480, -1024, -1056])
    ok1 = ok2 = True
    for _, Z in GAMMA:
        gz = int(G_VEC @ Z @ P_VEC)
        ok1 &= np.array_equal(B @ Z @ B, gz * outer(E_VEC, F_VEC))              # 4 N Z N = gamma e f^T
        for _, Zp in GAMMA:
            ok2 &= not (B @ Z @ B @ Zp @ B).any()                                # N Z N Z' N = 0
    check("(4): (2N) Z (2N) = gamma(Z) e f^T  (i.e. N Z N = gamma/4 e f^T) for all 16 Z", ok1)
    check("(4): N Z N Z' N = 0 for all 16 x 16 = 256 pairs (Z, Z') in Gamma", ok2)

    # ------------------------------------------------------------------ Section 5.2
    section("Section 5.2: Theorem 10 (all 64 translates, all 8 phases)")
    ok_a = ok_b = ok_c = True
    cnt = collections.Counter()
    for r in range(8):
        for c in range(8):
            X = um.window(M, r, c)
            f = 2 if (r - c) % 4 == 2 else 3
            ok_a &= um.ultimate_index(X) == f
            ok_b &= um.ultimate_index(X @ J) == 3
            im = um.d4_images(X)
            got = [um.ultimate_index(im[k]) for k in ("X", "X^T", "JXJ", "JX^TJ", "XJ", "JX", "X^TJ", "JX^T")]
            ok_c &= got == [f] * 4 + [3] * 4
            for Z in im.values():
                cnt[um.ultimate_index(Z)] += 1
    check("(1) nu(M_{r,c}) = 2 iff r-c = 2 mod 4, else 3", ok_a)
    check("(2) nu(M_{r,c} J) = 3 for all (r,c)", ok_b)
    check("(3) phases X, X^T, JXJ, JX^TJ have nu(1); phases XJ, JX, X^TJ, JX^T have nu = 3", ok_c)
    check("among the 512 squares of T u T^T: 64 have nu = 2 and 448 have nu = 3", cnt == {2: 64, 3: 448}, dict(cnt))
    check("M_{0,2} = Type II' has nu = 2", um.ultimate_index(IIb) == 2)

    # ------------------------------------------------------------------ Section 6 (products)
    section("Section 6: products")
    rnd = random.Random(2026)
    Tl = [np.frombuffer(b, dtype=np.uint8).reshape(8, 8).astype(np.int64) for b in sorted(T_set)]
    ok_t = all(np.array_equal(rnd.choice(Tl) @ rnd.choice(Tl) @ rnd.choice(Tl), 2197000 * E) for _ in range(5000))
    check("Theorem 13: 5000 random triples from T multiply to 2197000*E", ok_t)
    # Theorem 13, two factors: ALL 256 x 256 = 65536 pairs (A, B) in T x T (exhaustive)
    idx = [(ip, iq) for ip in range(16) for iq in range(16)]                       # A = P_ip M Q_iq
    Tarr = np.array([GAMMA[ip][1] @ M @ GAMMA[iq][1] for ip, iq in idx], dtype=np.int64)       # shape (256, 8, 8)
    prod = np.matmul(Tarr[:, None], Tarr[None, :])                                  # shape (256, 256, 8, 8)
    is_const = (prod == prod[..., :1, :1]).all(axis=(2, 3))
    S2, S6 = um.spow(2), um.spow(6)
    good_Z = [[bool(np.array_equal(GAMMA[iq][1] @ GAMMA[ip][1], S2) or np.array_equal(GAMMA[iq][1] @ GAMMA[ip][1], S6))
               for ip in range(16)] for iq in range(16)]                          # good_Z[iq][ip]: Q_A P_B in {S^2, S^6}
    expected = np.array([[good_Z[idx[a][1]][idx[b][0]] for b in range(256)] for a in range(256)])
    check("Theorem 13: for ALL 65536 pairs (A,B) in T x T:  AB is constant  iff  Q_A P_B in {S^2, S^6}",
          np.array_equal(is_const, expected) and int(is_const.sum()) == 8192, "(%d constant products)" % int(is_const.sum()))
    check("Theorem 13: every constant product AB equals 260^2/8 E = 8450 E",
          all(int(prod[a, b][0, 0]) == 8450 for a, b in zip(*np.nonzero(is_const))))
    # Theorem 13, three or more factors: the proof needs only  N Z N Z' N = 0  for all Z, Z' in Gamma;
    # (4) was checked above for N, here for N^T as well, which covers T^T (N_{P M^T Q} = P N^T Q).
    BT = B.T
    check("Theorem 13 / remark after Prop. 14: N^T Z N^T Z' N^T = 0 for all 256 pairs (Z, Z'), so triples from T^T are constant",
          all(not (BT @ Z @ BT @ Zp @ BT).any() for _, Z in GAMMA for _, Zp in GAMMA))
    C = J @ M.T
    D = M @ M @ C - 2197000 * E
    check("Proposition 14: M M C - 2197000 E = -1344 e (8e+p)^T != 0 for C = J M^T",
          np.array_equal(D, -1344 * outer(E_VEC, 8 * E_VEC + P_VEC)) and D.any())
    check("8e + p = (-7,-5,-3,-1,7,5,3,1)", (8 * E_VEC + P_VEC).tolist() == [-7, -5, -3, -1, 7, 5, 3, 1])
    ok_tt = all(np.array_equal(Tl[a].T @ Tl[b].T @ Tl[c].T, 2197000 * E)
                for a, b, c in (tuple(rnd.randrange(256) for _ in range(3)) for _ in range(2000)))
    check("products of three elements of T^T are also constant (2000 random triples)", ok_tt)

    # ------------------------------------------------------------------ Section 5.3
    section("Section 5.3: Sigma, Sigma', Planck's A-D method (Proposition 11)")
    sig = um.sigma_set()
    rho = um.rho_set()
    check("|Sigma| = |Sigma'| = 384 and Sigma' = Sigma^{-1}",
          len(set(sig)) == 384 and len(set(rho)) == 384 and set(rho) == {um.inverse_perm(s) for s in sig})
    check("Sigma = {sigma : sigma(7-i) = sigma(i)+4 mod 8} (brute force over 8!)",
          set(sig) == {p for p in itertools.permutations(range(8)) if all(p[7 - i] == (p[i] + 4) % 8 for i in range(8))})
    check("Sigma' = {rho : rho(a+4) = 7 - rho(a)} (brute force over 8!)",
          set(rho) == {p for p in itertools.permutations(range(8)) if all(p[(a + 4) % 8] == 7 - p[a] for a in range(8))})
    check("all 512 squares of T u T^T have the antipodal-complement property (5)",
          all(um.has_antipodal_complement(Z) for Z in S1.values()))
    check("Type I = (M^T)^(sigma0) with sigma0 = (0,1,2,3,7,6,5,4)", np.array_equal(um.permute(M.T, SIGMA0), TYPE_I))
    sols = [p for p in itertools.permutations(range(8)) if np.array_equal(um.permute(M.T, p), TYPE_I)]
    check("sigma0 is the only permutation with (M^T)^(sigma) = Type I", sols == [SIGMA0])
    W2 = um.perm_matrix(um.PLANCK_W2)
    W1 = um.perm_matrix(um.PLANCK_W1)
    check("sigma0 = Nordgren's W2 = [[I4,0],[0,R4]];  Type I = W2 M^T W2  (Planck's A-D method, [NOR12, Thm 9, Cor 10])",
          um.PLANCK_W2 == (0, 1, 2, 3, 7, 6, 5, 4) and np.array_equal(W2 @ M.T @ W2, TYPE_I))
    check("W1 = [[R4,0],[0,I4]] and W2 both lie in Sigma and in Sigma'; W1 M^T W1 is also a symmetric magic square",
          um.PLANCK_W1 in sig and um.PLANCK_W2 in sig and um.PLANCK_W1 in rho and um.PLANCK_W2 in rho
          and um.is_magic(W1 @ M.T @ W1) and um.is_associative(W1 @ M.T @ W1))
    check("Planck's A-D method (Nordgren Thm 9): W1 A W1 and W2 A W2 are pandiagonal magic squares for A = Type I",
          um.is_pandiagonal(W1 @ TYPE_I @ W1) and um.is_pandiagonal(W2 @ TYPE_I @ W2)
          and um.is_natural(W1 @ TYPE_I @ W1) and um.is_natural(W2 @ TYPE_I @ W2))
    check("Prop. 11 (i): all of X^(sigma), X in T u T^T, sigma in Sigma are natural symmetric magic squares with the same nu (sampled in 2000 cases)",
          all((lambda X, s: um.is_natural(um.permute(X, s)) and um.is_magic(um.permute(X, s)) and um.is_associative(um.permute(X, s))
               and um.ultimate_index(um.permute(X, s)) == um.ultimate_index(X))(rnd.choice(list(S1.values())), rnd.choice(sig))
              for _ in range(2000)))

    if args.quick:
        print("\n(--quick: exhaustive enumeration skipped)")
    else:
        enumerate_orbits(S1, sig, rho)
    passed, total = sum(RESULTS), len(RESULTS)
    print("\n%d / %d checks passed   (%.1f s)" % (passed, total, time.time() - t0))
    sys.exit(0 if passed == total else 1)


def enumerate_orbits(S1, sig, rho):
    """exhaustive part: 8! simultaneous permutations of the 512 squares, Sigma / Sigma' images, translates."""
    section("Section 5.3: exhaustive enumeration (counts of Table 4, Table 5 and the orbit statistics)")
    squares = [Z.astype(np.uint8).ravel() for Z in S1.values()]
    sq = np.array(squares)
    sq_lookup = {row.tobytes(): j for j, row in enumerate(sq)}                 # exact byte-string lookup (no hashing)
    idx_all = um.perm_index(np.array(list(itertools.permutations(range(8)))))
    unassigned = set(range(len(sq)))
    orbit_info = []
    assoc_members, pan_members = [], []
    while unassigned:
        i = min(unassigned)
        Y = sq[i][idx_all]                                                    # the 8! images of square i
        size = len(um.unique_rows(Y))
        members = sorted({sq_lookup[b] for b in (row.tobytes() for row in Y)
                          if b in sq_lookup and sq_lookup[b] in unassigned})
        if i not in members:
            raise RuntimeError("the orbit representative was not found among its own images")
        unassigned -= set(members)
        a = um.batch_associative(Y)
        p = um.batch_pandiagonal(Y)
        nu = um.ultimate_index(sq[i].reshape(8, 8).astype(np.int64))
        if not all(um.ultimate_index(sq[j].reshape(8, 8).astype(np.int64)) == nu for j in members):
            raise RuntimeError("nu is not constant on a similarity orbit")
        orbit_info.append((size, int(a.sum()), int(p.sum()), nu, len(members), int((a & p).sum())))
        assoc_members.append(Y[a])
        pan_members.append(Y[p])
    stats = collections.Counter(orbit_info)
    print("     (orbit size, #symmetric, #pandiagonal, nu, #squares of T u T^T in the orbit, #both) -> number of orbits")
    for k_, v_ in sorted(stats.items()):
        print("      ", k_, "->", v_)
    check("32 similarity orbits, each with 40320 distinct squares", len(orbit_info) == 32 and all(o[0] == 40320 for o in orbit_info))
    check("each orbit contains exactly 384 symmetric and 384 pandiagonal magic squares", all(o[1] == 384 and o[2] == 384 for o in orbit_info))
    check("each orbit contains 16 squares of T u T^T", all(o[4] == 16 for o in orbit_info))
    check("orbits with nu = 2: 4;  orbits with nu = 3: 28",
          sum(1 for o in orbit_info if o[3] == 2) == 4 and sum(1 for o in orbit_info if o[3] == 3) == 28)
    check("no square in these 32 orbits is both symmetric and pandiagonal", all(o[5] == 0 for o in orbit_info))

    # symmetric squares obtained by applying Sigma to the 512 squares
    idx_sig = um.perm_index(np.array(sig))
    acc = np.zeros((0, 64), dtype=np.uint8)
    for start in range(0, len(sq), 64):
        chunk = np.concatenate([sq[j][idx_sig] for j in range(start, min(start + 64, len(sq)))])
        acc = um.unique_rows(np.vstack([acc, um.unique_rows(chunk)]))
    assoc_orbit = um.unique_rows(np.concatenate(assoc_members))
    check("Sigma-images of the 512 squares: 12288 distinct symmetric magic squares, equal to the symmetric members of the 32 orbits",
          len(acc) == 12288 and np.array_equal(acc, assoc_orbit) and um.batch_associative(acc).all())
    nu_assoc = um.batch_nu_le3(acc)
    check("nu of the 12288 symmetric squares: 10752 with nu = 3 and 1536 with nu = 2",
          collections.Counter(nu_assoc.tolist()) == {3: 10752, 2: 1536}, dict(collections.Counter(nu_assoc.tolist())))

    # D4 classes of the symmetric squares
    def canon(row):
        X = row.reshape(8, 8).astype(np.int64)
        return min(Z.astype(np.uint8).tobytes() for Z in um.d4_images(X).values())

    classes = collections.defaultdict(set)
    for row, nu in zip(acc, nu_assoc):
        classes[int(nu)].add(canon(row))
    check("D4 classes of the symmetric squares: 1344 (nu = 3) + 192 (nu = 2) = 1536 = 12288/8",
          {k: len(v) for k, v in classes.items()} == {3: 1344, 2: 192}, {k: len(v) for k, v in classes.items()})

    # pandiagonal squares obtained by applying Sigma' to the symmetric squares
    idx_rho = um.perm_index(np.array(rho))
    pan_all = np.zeros((0, 64), dtype=np.uint8)
    all_pandiag = True
    for start in range(0, len(acc), 64):
        chunk = np.concatenate([acc[j][idx_rho] for j in range(start, min(start + 64, len(acc)))])
        all_pandiag &= bool(um.batch_pandiagonal(chunk).all())
        pan_all = um.unique_rows(np.vstack([pan_all, um.unique_rows(chunk)]))
    pan_orbit = um.unique_rows(np.concatenate(pan_members))
    check("Prop. 11 (ii): all 12288 x 384 images under Sigma' are pandiagonal magic squares (exhaustive)", all_pandiag)
    check("Sigma'-images of the symmetric squares: 12288 distinct pandiagonal squares = the pandiagonal members of the 32 orbits",
          len(pan_all) == 12288 and np.array_equal(pan_all, pan_orbit))
    check("the 512 squares of T u T^T are among them",
          len({r.tobytes() for r in pan_all} & {r.tobytes() for r in sq}) == 512)
    nu_pan = um.batch_nu_le3(pan_all)
    check("nu of the 12288 pandiagonal squares: 10752 with nu = 3 and 1536 with nu = 2",
          collections.Counter(nu_pan.tolist()) == {3: 10752, 2: 1536})

    # translates (Table 5)
    cnt = collections.Counter()
    for row in pan_all:
        Bt = 2 * row[um.TRANS_IDX].astype(np.int64).reshape(64, 8, 8) - um.COMPL
        z3 = ((Bt @ Bt @ Bt) == 0).all(axis=(1, 2))
        cnt[int(z3.sum())] += 1
    print("     number of ultimate translates (of 64) -> number of pandiagonal squares:", dict(sorted(cnt.items())))
    check("Table 5: 32 -> 3584, 48 -> 512, 64 -> 8192 (these three cases exhaust the 12288 squares)",
          dict(cnt) == {32: 3584, 48: 512, 64: 8192})


if __name__ == "__main__":
    main()
