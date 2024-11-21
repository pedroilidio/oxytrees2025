from pathlib import Path
import pandas as pd
import numpy as np

BASE_URL = 'http://staff.cs.utu.fi/~aatapa/data/DrugTarget/'
THRESH = 30
SEED = 0

print('Fetching drug similarities...')
X1 = pd.read_table(
    BASE_URL + 'drug-drug_similarities_ECFP4.txt',
    header=None,
    sep='\\s+',
)
print('Fetching target similarities...')
X2 = pd.read_table(
    BASE_URL + '/target-target_similarities_WS_normalized.txt',
    header=None,
    sep='\\s+',
)
X2 /= 100  # Normalize to 0-1 interval

print('Fetching affinity scores...')
y = pd.read_table(
    BASE_URL + 'drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt',
    header=None,
    sep='\\s+',
)
breakpoint()

out_dir = Path('datasets')
out_dir.mkdir(exist_ok=True)

print('Saving downloaded tables...')
X1.to_csv(out_dir/'X1.txt', header=None, index=False, sep='\t')
X2.to_csv(out_dir/'X2.txt', header=None, index=False, sep='\t')
y.to_csv(out_dir/'y.txt', header=None, index=False, sep='\t')

# Negative interactions are those with the smallest dissociation constants.
y_bin_mask = y < THRESH
y_bin = y_bin_mask.astype(float)
n_positives = y_bin_mask.values.sum()
index_positives = np.where(y_bin_mask)

rng = np.random.default_rng(SEED)
discovery_prob = np.arange(1, 0, -0.1)

for p in discovery_prob:
    print(f'Saving y with {p * 100:.0f}% of positive values...')
    positives_to_mask = rng.choice(
        n_positives,
        size=int((1 - p) * n_positives),
        replace=False,
    )
    indices_to_mask = (
        index_positives[0][positives_to_mask],
        index_positives[1][positives_to_mask],
    )

    y_sample = y_bin.copy()
    y_sample.values[indices_to_mask] = 0.0

    y_sample.to_csv(
        out_dir/f'y{p * 100:.0f}.txt',
        header=None,
        index=False,
        sep='\t',
    )

print('Done.')
