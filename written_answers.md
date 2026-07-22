# Written Answers — MNIST PCA & GMM Project

> These are draft answers for the **written** parts of `project.ipynb` (Part 7 analysis
> and the four Part 8 reflection questions). Read them, tweak the wording to sound like
> you, and paste them into the matching markdown cells in the notebook.
>
> ✅ **Updated with your actual run.** The numbers below (test accuracy 96.05%, best
> config n_pca=50/K=2 → 96.25%, most-confused pairs 3↔8 and 4↔9) are taken directly
> from the outputs in your executed notebook.

---

## Part 7 — Hyperparameter Study: Analysis

*(Goes in cell 42, replacing "Write your analysis here")*

Increasing the number of **PCA components** helped up to a point and then hurt. Accuracy rose
from ~95.5% at 20 components to a peak of **96.25% at 50 components**, but then *fell* to ~94.7%
at 80 components. The first 50 components capture the discriminative shape information, while the
extra dimensions past that mostly add noise and force each GMM to estimate a much larger
covariance matrix from the same amount of data — so the model becomes harder to fit reliably and
accuracy drops.

Increasing the number of **GMM components (K)** per class did **not** help in my grid — it actually hurt slightly. At the best PCA setting (50 dims), K=2 gave 96.25%, K=4 gave 96.00%, and K=8 dropped to 95.30%. With only ~1,000 training images per digit, adding more Gaussians splits the data too thinly: each component is estimated from fewer points, its covariance becomes less reliable, and the model starts to overfit rather than capture genuinely new writing styles.

So yes, there is a clear point of diminishing returns and even a measurable **decrease** in accuracy in both directions. The best configuration was **n_pca=50, K=2 → 96.25%**, which balances expressive power against having enough data to estimate each covariance matrix reliably. (For reference, the default n_pca=50, K=4 model in Part 6 scored      96.05%, versus 78.8% for K-Means K=1 and 87.7% for K-Means K=4 — the GMM comfortably beats both baselines.)

---

## Q1 — PCA & the Curse of Dimensionality

*(Goes in cell 44)*

Fitting a GMM directly in the full 784-D pixel space fails for two concrete reasons:

**1. The covariance matrices become singular / non-invertible.**
Each component needs a 784×784 covariance matrix, which has about 784·785/2 ≈ 307,000 free parameters. With only ~1,000 training images per digit, there are far fewer data points than parameters, so the estimated covariance matrix is **rank-deficient (singular)**. The Gaussian log-pdf requires `Σ⁻¹` and `log|Σ|`; for a singular matrix the determinant is ~0 and the inverse is undefined, so the whole likelihood computation blows up (NaNs / infinities) or is wildly unstable. On top of that, many MNIST pixels (the corners/borders) are always 0, giving those dimensions exactly zero variance — an immediate singularity.

**2. The EM algorithm becomes computationally expensive and gets stuck.**
Every E-step evaluates the multivariate Gaussian in 784-D, and every M-step re-estimates and inverts a 784×784 matrix per component per iteration — an `O(D³)` cost that is enormous compared to working in 50-D. Beyond the raw cost, in such high dimensions distances between points become nearly uniform (the curse of dimensionality), so responsibilities are poorly separated and EM converges slowly to poor local optima. PCA fixes both problems at once: reducing to ~50 dimensions gives well-conditioned, invertible covariance matrices that can actually be estimated from the available data.

---

## Q2 — The Log-Sum-Exp Trick

*(Goes in cell 45)*

In the E-step we need `log Σ_k π_k N(x | μ_k, Σ_k)`, but in ~50-D the individual log-terms are large negative numbers. Suppose for one point the three log-terms are:

```
ℓ = [-1000, -1002, -1005]
```

**Naive approach — exponentiate, then sum, then take the log:**
`exp(-1000)`, `exp(-1002)`, `exp(-1005)` are all far below the smallest number `float64` can represent (~1e-308), so each one **underflows to exactly 0.0**. Summing gives `0.0`, and `log(0.0) = -inf`. The information is completely lost, and any downstream division to normalise the responsibilities becomes `0/0 = NaN`.

**Log-sum-exp trick — factor out the maximum first:**
Let `m = max(ℓ) = -1000`. Then

```
log Σ_k exp(ℓ_k) = m + log Σ_k exp(ℓ_k - m)
```

The shifted terms `ℓ - m = [0, -2, -5]` exponentiate to `[1.0, 0.135, 0.0067]`, which are
perfectly safe to sum → `1.142`. So the result is `-1000 + log(1.142) ≈ -999.87`, a correct, finite answer. The largest term (which dominates the sum anyway) is always exactly `exp(0)=1`, so it never underflows, and the trick recovers the true log-sum without any overflow or underflow.

---

## Q3 — Confusion Matrix

*(Goes in cell 46)*

The two digit pairs my classifier confuses most often are **3 ↔ 8** and **4 ↔ 9**. In my
confusion matrix, 3↔8 was by far the largest off-diagonal pair (13 misclassifications combined — 8 threes predicted as eights and 5 eights predicted as threes), followed by 4↔9 (7 combined). A few others were close behind (7↔9, 3↔5, 1↔8, each around 6).

These digits share most of their pixel-level structure. An 8 is essentially a 3 with the
left side of its loops closed, so a rounded or overlapping 3 looks almost identical to an 8. A 4 and a 9 differ mainly in whether the top loop is closed. After projecting to 50 PCA dimensions — which capture broad shape rather than fine local strokes — those small distinguishing details (one open vs. closed curve) are largely smoothed away, so the per-class Gaussians for these digits overlap heavily and the MAP rule picks the wrong one on the ambiguous, sloppily-written examples. The low precision on digit 8 (0.89) in the classification report confirms that several other digits are being pulled into the "8" class.

Keep more PCA components (or use features that preserve local stroke detail) so the distinguishing curves/loops survive dimensionality reduction — or increase
the number of GMM components per class so each digit's distinct writing styles are modelled separately instead of being merged into one overlapping blob.

---

## Q4 — GMM vs. Discriminative Models

*(Goes in cell 47)*

Two fundamental reasons the generative GMM is at a disadvantage versus a CNN (beyond parameter
count):

**1. It optimises the wrong objective — modelling `p(x|c)` instead of the decision boundary.**
The GMM is *generative*: each class's GMM is trained only on that class's own data to model how its images are distributed, with no knowledge of the other classes. It spends its capacity describing *what a digit looks like* rather than *what separates one digit from another*. A CNN is *discriminative*: it directly optimises `p(c|x)`, so every parameter is tuned to place decision boundaries between classes. For classification, modelling the boundary directly is far more efficient than accurately modelling each full class density and hoping the boundaries fall out correctly.

**2. It works on flattened pixels and cannot exploit spatial structure / invariances.**
The GMM (even after PCA) treats an image as an unstructured 784- (or 50-) dimensional vector, so a digit shifted, rotated, or scaled by a few pixels looks like a completely different point. A CNN uses convolutions and pooling to build **translation-invariant, hierarchical features** (edges - strokes - shapes), so it recognises the same digit regardless of small spatial variations. The GMM has no built-in notion that neighbouring pixels are related or that a shifted digit is the same digit, which caps how well it can generalise on image data.

---

*End of draft answers — edit freely before pasting into the notebook.*
