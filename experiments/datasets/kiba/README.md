# The KIBA datset

The KIBA dataset was initially built by [Tang et al., 2014](https://pubs.acs.org/doi/10.1021/ci400709d) and contains experimentally verified affinity scores between kinase and kinase inhibitors.

[He et al., 2017](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z) further processed the dataset as described by the authors:

> \[The KIBA dataset was preprocessed\] by removing all drugs and targets with less than 10 observations from the original dataset (downloaded from the supplementary materials of [42]), resulting in a dataset of 2116 drugs and 229 targets with a density of 24%.

> In the KIBA dataset, the lower the KIBA-score, the higher the binding affinity, and [42] suggests a threshold of KIBA value ≤3.0 to binarize the dataset. In an additional preprocessing step, we transformed the KIBA dataset by taking the negative of each value and adding the minimum to all values in order to obtain a threshold where all values above the threshold are classified as binding. The KIBA threshold of 3.0 in the untransformed dataset then becomes 12.1. 

It contains 2111 drugs and 229 proteins, and, after applying the binarization threshold, it has a density of 19.74% (non-missing entries in general have density of 24.4%).

Finally, the dataset, with corresponding amino acid sequences and SMILES representations were provided by [Huang et al., 2020](https://academic.oup.com/bioinformatics/article/36/22-23/5545/6020256).

To fetch, process the dataset and generate similarity matrices, run the following command:

```bash
python get_kiba.py
```

The final files will be saved at the `final` directory.

The protein similarities are the (normalized) Smith-Waterman scores between the protein sequences, using no gap penalty and the BLOSUM62 as the substitution matrix.

The compound similarity values are the Tanimoto similarity between the Morgan fingerprints of the compounds, using a radius of 2 (a.k.a. the Extended Compound Fingerprints or ECFP4).