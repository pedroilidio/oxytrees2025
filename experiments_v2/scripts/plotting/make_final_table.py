from pathlib import Path

import pandas as pd
import click


def make_final_table(table_path: list[Path]) -> pd.DataFrame:
    all_tables = pd.concat((pd.read_csv(p) for p in table_path))
    all_tables = all_tables.loc[all_tables.dataset != "all_datasets"]
    all_tables[["dataset", "validation_setting"]] = all_tables.dataset.str.split(
        "__", expand=True, n=1
    )
    all_tables = all_tables.loc[:, ~all_tables.columns.str.contains("rank")]
    all_tables = all_tables.loc[:, ~all_tables.columns.str.contains("victories")]

    # XXX: remove after fixing upstream
    all_tables["dataset"] = all_tables.dataset.str.replace("_", " ")
    all_tables["validation_setting"] = all_tables.validation_setting.str.replace("_", " ")

    index_cols = ["dataset", "validation_setting", "estimator"]

    auroc_inductive = (
        all_tables.set_index(index_cols)
        .loc[:, "AUROC (Inductive)"]
        .unstack(level=["validation_setting"])
        .apply(lambda col: col.replace(r"<b>(.+?)</b>", r"\\textbf{\1}", regex=True))
    )
    breakpoint()
    return all_tables


@click.command()
@click.option(
    "--table-path",
    "-t",
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    required=True,
    multiple=True,
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
def main(table_path, output_path):
    final_table = make_final_table(table_path)
    print(final_table)
    final_table.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
