important ref
* 23, 28

Data
* format: class + time_seq
* TODO: HIVE-COTE, z-normalization
* time series of different lengths:
  * pad NaN for length consistency in storing data
  * add low amplitude random numbers in computing baseline
* time series with missing value
  * NaN in storing data
  * linear interpolation in computing baseline

Dynamic time warping
* `dist, ix, iy = dtw(x, y, w)`
* How to choose a proper size of warping window?
  * The best warping window width is decided by performing Leave-One-Out Cross-Validation (LOO CV) with the train set, choosing the smallest value of $w$ that minimizes the average train error rate.
  * Notice it sometimes can produce poor results, the best $w$ in training may not be the best $w$ for testing. **This issue is common in real world deployments.**
* just smoothing the data can produce significant improvements

wavelet representation (multi-resolution?)
* authors suppose that wavelet representation is simply smoothing the data implicitly
* Haar wavelet transformation, Piecewise Aggerate Approximation are similar to the moving average filter smoothing
* All is attribute to the SMOOTHING!
* ? Why smoothing so works

Lack of ablation to forcefully convince us
* exploitation of the memory of HMM
* long-term dependency features of CNN

Fairness
* data classifier: 1-NN
* data split: resampling while remaining the default train and test splits
* Authors advise using a Wilcoxon signed-rank test with a significance level of $\alpha\leq0.05$ to check if there are significant differences between the classifiers