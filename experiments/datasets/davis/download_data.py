import pandas as pd
from pathlib import Path
from DeepPurpose import dataset, utils
import data_process

# drug_indices_url = \
#     'http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug_PubChem_CIDs.txt'
# target_indices_url = \
#     'http://staff.cs.utu.fi/~aatapa/data/DrugTarget/target_gene_names.txt'
# 
# X_target_indices = pd.read_csv(target_indices_url)
# X_drug_indices = pd.read_csv(drug_indices_url)

dir_data = Path('./data/')

X_drug, X_target, affinity = dataset.load_process_DAVIS(
    str(dir_data),
    binary=False,
)

X_target = pd.read_json(dir_data/'DAVIS/target_seq.txt', typ='series')
X_drug = pd.read_json(dir_data/'DAVIS/SMILES.txt', typ='series')

X_drug.to_csv(dir_data/'DAVIS/SMILES.csv', index=False, header=None)
X_target.to_csv(dir_data/'DAVIS/target_seq.csv', index=False, header=None)

# affinity = pd.read_csv(
#     dir_data/'DAVIS/affinity.txt',
#     sep = ' ',
#     names=X_target.index,
# )
# affinity.index = X_drug.index
