import warnings

import numpy as np
import pandas as pd
from DeepPurpose.utils import encode_drug, encode_protein, convert_y_unit


def preprocess_dti_for_deeppurpose(
    X,
    Y,
    drug_encoding,
    target_encoding,
    threshold=None,
    convert_to_log=False,
    subset=None,
):
    X_drug, X_target = X
    assert X_drug.shape[0] == X_drug.size
    assert X_target.shape[0] == X_target.size
    assert Y.shape == (X_drug.size, X_target.size)

    if isinstance(Y, (pd.DataFrame, pd.Series)):
        # TODO: more specific conditional on X
        Y = Y.loc[X_drug.index, X_target.index].values
    if isinstance(X_drug, (pd.DataFrame, pd.Series)):
        X_drug = X_drug.values
    if isinstance(X_target, (pd.DataFrame, pd.Series)):
        X_target = X_target.values

    df_data = pd.DataFrame({
        'SMILES': np.repeat(X_drug.reshape(-1), X_target.size),
        'Target Sequence': np.tile(X_target.reshape(-1), X_drug.size),
        'Label': Y.reshape(-1),
    })
    if subset is not None:
        df_data = df_data.iloc[subset]

    is_finite = np.isfinite(df_data.Label)

    if not is_finite.all():
        warnings.warn(
            f"Found {(~is_finite).sum()}/{len(df_data)} non-finite"
            f" labels (nan or inf). They will be removed. {is_finite.sum()}"
            " entries remaining."
        )
        df_data = df_data.loc[is_finite]

    if threshold is not None:
        df_data['Label'] = (df_data['Label'] < threshold).astype(float)
    elif convert_to_log:
        df_data['Label'] = convert_y_unit(df_data['Label'], 'nM', 'p')

    df_data = encode_drug(df_data, drug_encoding)
    df_data = encode_protein(df_data, target_encoding)

    return df_data.reset_index(drop=True)

