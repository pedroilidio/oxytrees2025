# Oxytrees: Model Trees for Bipartite Learning

**Authors:**
Pedro Ilídio \[1, 2\]*, Felipe Kenji Nakano \[1, 2\], Alireza Gharahighehi \[1, 2\],
Robbe D’hondt \[1, 2\], Ricardo Cerri \[3\], Celine Vens \[1, 2\]

**Affiliations:**
- \[1\] Dept. of Public Health and Primary Care, KU Leuven, Campus
KULAK, Etienne Sabbelaan 53, Kortrijk, 8500, Belgium.
- \[2\] Itec, imec research group at KU Leuven, Etienne Sabbelaan 51,
Kortrijk, 8500, Belgium.
- \[3\] Instituto de Ciências Matemáticas e de Computação, Universidade de
São Paulo, São Carlos, Av. Trab. São Carlense, São Carlos, 13566-590,
São Paulo, Brazil.

**\*Corresponding author e-mail:** [pedro.ilidio@kuleuven.be](mailto:pedro.ilidio@kuleuven.be)

**Resources:**
- `bipartite_learn` Python package: [https://bipartite-learn.readthedocs.io](https://bipartite-learn.readthedocs.io)
- arXiv extended version: [https://arxiv.org/abs/2511.12713](https://arxiv.org/abs/2511.12713)

## Abstract

Bipartite learning is a machine learning task that aims to
predict interactions between pairs of instances. It has been
applied to various domains, including drug-target interactions,
RNA-disease associations, and regulatory network inference.
Despite being widely investigated, current methods
still present drawbacks, as they are often designed for a specific
application and thus do not generalize to other problems
or present scalability issues. To address these challenges,
we propose Oxytrees: proxy-based biclustering model
trees. Oxytrees compress the interaction matrix into row- and
column-wise proxy matrices, significantly reducing training
time without compromising predictive performance. We also
propose a new leaf-assignment algorithm that significantly reduces
the time taken for prediction. Finally, Oxytrees employ
linear models using the Kronecker product kernel in their
leaves, resulting in shallower trees and thus even faster training.
Using 15 datasets, we compared the predictive performance
of ensembles of Oxytrees with that of the current stateof-
the-art. We achieved up to 30-fold improvement in training
times compared to state-of-the-art biclustering forests,
while demonstrating competitive or superior performance in
most evaluation settings, particularly in the inductive setting.
Finally, we provide an intuitive Python API to access all
datasets, methods and evaluation measures used in this work,
thus enabling reproducible research in this field.

**Keywords:** bipartite learning, biclustering trees, model trees, regularized least
squares, positive-unlabeled learning