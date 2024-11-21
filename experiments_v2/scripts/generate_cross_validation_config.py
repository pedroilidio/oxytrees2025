from functools import cached_property
from itertools import product
from dataclasses import dataclass

import click
import yaml
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.utils import check_random_state

# TODO: only test indices: enforce non-overlapping indices
# TODO: enforce consistent dataset definitions (maybe dataclass or pydantic)
# TODO: add another file with instance names to the datasets


@dataclass
class FoldRecord:
    test_rows: list[int]
    test_cols: list[int]
    masked_positives: list[int]


class TransductiveFoldGenerator:
    def __init__(self, n_splits, random_state, y):
        self.n_splits = n_splits
        self.random_state = random_state
        self.positive_indices = np.flatnonzero(y)

        self.test_rows = []
        self.test_cols = []

    def __iter__(self):
        cv_splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for _, test_indices in cv_splitter.split(self.positive_indices):
            yield FoldRecord(
                test_rows=self.test_rows,
                test_cols=self.test_cols,
                masked_positives=self.positive_indices[test_indices].tolist(),
            ).__dict__  # TODO: get pyyaml to do this


class InductiveFoldGenerator:
    def __init__(
        self, *, n_row_splits, n_col_splits, random_state, y, positive_masking_percent=0
    ):
        self.n_row_splits = n_row_splits
        self.n_col_splits = n_col_splits
        self.random_state = random_state
        self.positive_masking_percent = positive_masking_percent
        self.n_rows, self.n_cols = y.shape
        self.y = y
        self.y_vec = y.reshape(-1)

        # Row and column indices in the flattened y
        self.row_groups = np.repeat(np.arange(self.n_rows), self.n_cols)
        self.col_groups = np.tile(np.arange(self.n_cols), self.n_rows)

    @cached_property
    def _rng(self):
        return check_random_state(self.random_state)

    def generate_masked_positives(self, y_train):
        if self.positive_masking_percent == 0:
            return []
        positive_indices = np.flatnonzero(y_train)
        mask_size = int(self.positive_masking_percent / 100 * positive_indices.size)
        mask = self._rng.choice(positive_indices, size=mask_size, replace=False)
        return mask.tolist()

    def iter_row_splits(self):
        yield from iter_axis_splits(
            y_vec=self.y_vec,
            n_splits=self.n_row_splits,
            n_samples=self.n_rows,
            sample_groups=self.row_groups,
            random_state=self.random_state,
            shuffle=True,
        )

    def iter_col_splits(self):
        yield from iter_axis_splits(
            y_vec=self.y_vec,
            n_splits=self.n_col_splits,
            n_samples=self.n_cols,
            sample_groups=self.col_groups,
            random_state=self.random_state,
            shuffle=True,
        )

    def __iter__(self):
        for (train_rows, test_rows), (train_cols, test_cols) in product(
            self.iter_row_splits(), self.iter_col_splits()
        ):
            yield FoldRecord(
                test_rows=test_rows,
                test_cols=test_cols,
                masked_positives=self.generate_masked_positives(
                    self.y[train_rows, :][:, train_cols]
                ),
            ).__dict__  # TODO: get pyyaml to do this


def iter_axis_splits(
    *, y_vec, n_splits, n_samples, sample_groups, random_state, shuffle
):
    if n_splits == 1:
        yield (list(range(n_samples)), [])
        return

    cv_splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    for train_indices, test_indices in cv_splitter.split(
        y_vec, y_vec, groups=sample_groups
    ):
        axis_train_indices = np.unique(sample_groups[train_indices])
        axis_test_indices = np.unique(sample_groups[test_indices])

        yield axis_train_indices.tolist(), axis_test_indices.tolist()


@click.command()
@click.option(
    "--dataset-definitions",
    type=click.File("r"),
    required=True,
    help="YAML file with dataset definitions.",
)
@click.option(
    "--cross-validation-config",
    type=click.File("r"),
    required=True,
    help="YAML file with cross-validation configuration.",
)
@click.option(
    "--fold-definitions",
    type=click.File("w"),
    required=True,
    help="Output YAML file with cross-validation configuration.",
)
def main(
    dataset_definitions,
    cross_validation_config,
    fold_definitions,
):
    """Generate cross-validation configuration for each dataset."""
    dataset_definitions = yaml.safe_load(dataset_definitions)
    cross_validation_config = yaml.safe_load(cross_validation_config)
    out = {}

    for dataset_name in dataset_definitions.keys():
        dataset_config = dataset_definitions[dataset_name]
        cv_config = cross_validation_config[dataset_name]
        random_state = cv_config["random_state"]

        print("- Processing dataset:", dataset_name)
        y = np.load(dataset_config["y"])

        out[dataset_name] = {}
        cv_settings = (
            (name, n_splits)
            for name, n_splits in cv_config.items()
            if name in ("TT", "TL", "LT", "LL")
        )
        if not cv_settings:
            raise ValueError("No valid cross-validation settings found.")

        for cv_name, n_splits in cv_settings:
            if cv_name == "LL":
                fold_generator = TransductiveFoldGenerator(
                    n_splits=n_splits, random_state=random_state, y=y
                )
                out[dataset_name][cv_name] = list(fold_generator)
                continue

            for masking_percent in cv_config["positive_masking_percent"]:
                if cv_name == "TT":
                    fold_generator = InductiveFoldGenerator(
                        n_row_splits=n_splits,
                        n_col_splits=n_splits,
                        random_state=random_state + 1,  # Different seed from LT and TL
                        y=y,
                        positive_masking_percent=masking_percent,
                    )
                elif cv_name == "LT":
                    fold_generator = InductiveFoldGenerator(
                        n_row_splits=1,
                        n_col_splits=n_splits,
                        random_state=random_state,
                        y=y,
                        positive_masking_percent=masking_percent,
                    )
                else:  # cv_name == "TL"
                    fold_generator = InductiveFoldGenerator(
                        n_row_splits=n_splits,
                        n_col_splits=1,
                        random_state=random_state,
                        y=y,
                        positive_masking_percent=masking_percent,
                    )

                if masking_percent > 0:
                    final_cv_name = f"{cv_name}_{masking_percent}"
                else:
                    final_cv_name = cv_name

                out[dataset_name][final_cv_name] = list(fold_generator)

    yaml.dump(out, fold_definitions)


if __name__ == "__main__":
    main()
