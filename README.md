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

**\*Corresponding author e-mail:** pedro.ilidio@kuleuven.be

## Abstract
Bipartite learning is a machine learning task aimed at predicting interactions among
pairs of instances. Several applications have been addressed, such as drugtarget
interaction, RNA-disease association and regulatory network inference.  Despite widely
investigated, current methods still present drawbacks, as they are often designed for a
specific application and thus do not generalize to other problems, or present
scalability issues. To address these challenges, we propose Oxytrees: efficient
biclustering model trees. More specifically, Oxytrees use novel algorithms for induction
and inference, leading to a complexity improvement proportional to the logarithm of the
number of pairs. Further, Oxytrees employ linear models using the Kronecker product
kernel (RLS-Kron) as their leaf models.  Using 15 datasets, we compared the predictive
performance of ensembles of Oxytrees against the most prominent methods from the
literature. Our results highlight that our method yields competitive or superior
performance in most of the cases, especially in the inductive setting, alongside the
substantial improvements in computational complexity. Finally, we propose a novel Python
library, bipartite learn, a simple and accessible tool that includes all datasets,
methods and evaluation measures used in this work, thus enabling reproducible research
in this field.

**Keywords:** bipartite learning, biclustering trees, model trees, regularized least
squares, positive-unlabeled learning