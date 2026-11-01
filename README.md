# ultimate-magic8

Verification code for the paper "The ultimate index of 8th-order magic squares: behaviour under rotation, translation and products" (original title in Japanese: 8次魔方陣の究極指数).
日本語版: [README（日本語版）.md](README（日本語版）.md)

| File | Contents |
|---|---|
| `ultimate_magic8.py` | The square data and exact integer-arithmetic tools (including small helpers for order 4) |
| `verify_paper.py` | A verification program for the paper's algebraic identities, examples, and numerical enumerations |

Put the two files in the **same folder**. All you need is Python 3 and `numpy`.

---

## Run it in Google Colab (no installation, beginner friendly)

All you need is a Google account and a browser; nothing has to be installed on your computer.

### Option A: fetch the code from GitHub

1. Open <https://colab.research.google.com> and choose "New notebook".
2. Paste the following into the first cell and press the ▶ button at its left (replace `<account>/<repository>` with the real name).

   ```python
   !git clone https://github.com/<account>/<repository>.git
   %cd <repository>
   ```

3. Run the verification in the next cell (about 20 seconds).

   ```python
   !python verify_paper.py
   ```

   If you only want a quick answer, add `--quick`, which skips the exhaustive enumerations (about 1 second).

   ```python
   !python verify_paper.py --quick
   ```

### Option B: upload the files directly (if you do not use GitHub)

1. In a new notebook, run:

   ```python
   from google.colab import files
   files.upload()
   ```

2. In the file dialog, select **both** `ultimate_magic8.py` and `verify_paper.py` at once and upload them.
3. Run in the next cell:

   ```python
   !python verify_paper.py
   ```

### Common pitfalls in Colab

- **Run the script with `!python`.** `import ultimate_magic8` looks for `ultimate_magic8.py` in the same folder as the script. In Option A you must first move into the folder with `%cd <repository>`.
- If one of the two files is missing you will see `ModuleNotFoundError: No module named 'ultimate_magic8'`. Check with `!ls` that both files are in the current folder.
- `numpy` is preinstalled in Colab, so no `pip install` is needed.
- Colab sessions expire after a while and uploaded files disappear; in that case start again from the first cell.

---

## Run it with your own Python

```bash
pip install numpy
python verify_paper.py            # everything (about 20 seconds)
python verify_paper.py --quick    # skip the exhaustive enumerations (about 1 second)
```

## Reading the output

- Every check prints `[PASS]` or `[FAIL]`.
- The run succeeds if the last line reads `74 / 74 checks passed` (about 20 seconds on Colab, depending on the machine); the exit status is then 0.
- With `--quick` the last line reads `61 / 61 checks passed`. Only the exhaustive enumerations (the 32 similarity orbits, Tables 4 and 5 and the orbit statistics) are skipped.
- If any check fails, the exit status is 1.

## Correspondence with the paper

The labels are the numbers printed in the paper as compiled from its TeX source, where definitions, lemmas, propositions, theorems, corollaries and remarks share **one** counter.

| No. | Content | No. | Content |
|---|---|---|---|
| Lemma 1 | power formula | Lemma 9 | inner products (Table 3) |
| Proposition 4 | rank 3 implies index 2 or 3 | Theorem 10 | translations and the ultimate index |
| Remark 5 | the Hill–Lettington–Schmidt square | Proposition 11 | maps by permutations (384 of them, Planck's method) |
| Theorem 6, Corollary 7 | symmetric squares and $D_4$ | Theorem 13 | products |
| Remark 8 | the order-4 phenomenon (F₁₀₉) | Proposition 14 | counterexample for products |

Equations: (1) $NE=EN=0$, (2) $JNJ=-N$, (3) $2N=eg^{\top}+pf^{\top}$, (4) $NZN=\frac14\gamma e f^{\top}$, (5) antipodal complement property.

## What exactly is checked

- For two factors, Theorem 13 is checked on **all 65,536 pairs** in $T\times T$. For three or more factors the program checks, for all 16×16 pairs in $\Gamma$, the identity $NZNZ'N=0$ used in the proof (and its transposed version). Random sampling is used only as an additional sanity check. Proposition 11(i) is additionally tested on 2,000 sampled pairs of squares and permutations; the full run exhaustively enumerates the similarity orbits and all images under Sigma and Sigma'. The index-preservation statement also follows mathematically from similarity.
- Remark 8 (order 4) is checked on **all 384** pandiagonal magic squares of order 4, enumerated exhaustively as solutions of linear conditions. Exactly 64 of them have $\nu=\infty$ with three nonzero eigenvalues while their 90° rotation has $\nu=3$ with 34 as the only nonzero eigenvalue. F₁₀₉ itself (Loly et al. 2009, equation (37)) is built into `ultimate_magic8.py`; the program checks that it is one of the 384 enumerated squares, that it has $\nu=\infty$, and that its rotation has $\nu=3$ with characteristic polynomial $x^3(x-34)$.

## Notes

- All arithmetic is exact integer arithmetic (the centred matrix N has half-integer entries, so 2N = 2A − 65E is used).
- The square data were typed in from the paper; checks such as the characteristic polynomials also detect typing errors.
- If you change the way theorems are numbered in the TeX source (for example per section), update the explanation and the labels at the top of `verify_paper.py`.

## Citation and licence

(To be filled in when published: DOI of the repository, e.g. via Zenodo, and licence.)
