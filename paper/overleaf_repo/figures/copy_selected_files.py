from pathlib import Path
import shutil

HERE = Path(__file__).parent.resolve()

IN = HERE.parents[2] / "experiments/results"
OUT = HERE / "results"
OUT_CAMERA_READY = HERE / "../camera_ready"

GLOB_PATTERNS = (
    "*/*/all_datasets/critical_difference_diagrams/*.pdf",
    "latex_tables/0/*.tex",
    "latex_tables/0/*.pdf",
    "empirical_complexity_plots/*.pdf",
    "ablation_relative/*.pdf",
    "leaf_size/*.pdf",
    "number_of_trees/*.pdf",
    "*/*/parallel_coordinates/*.pdf",
)

# These are also copied to the camera-ready folder (no subfolders allowed in the submitted source code)
CAMERA_READY_FIGURES = {
    "ablation_Inductive_AUPRC_50.pdf": "ablation_relative/Inductive_AUPRC_50.pdf",
    "ablation_Inductive_AUPRC_00.pdf": "ablation_relative/Inductive_AUPRC_00.pdf",
    "baselines_AUPRC_Inductive.pdf": (
        "literature_methods"
        "/TT/all_datasets"
        "/critical_difference_diagrams"
        "/AUPRC (Inductive).pdf"
    ),
    "baselines_AUROC_Inductive.pdf": (
        "literature_methods"
        "/TT/all_datasets"
        "/critical_difference_diagrams"
        "/AUROC (Inductive).pdf"
    ),
    "leaf_size_Inductive_AUPRC_00.pdf": "leaf_size/Inductive_AUPRC_00.pdf",
    "leaf_size_Inductive_AUROC_00.pdf": "leaf_size/Inductive_AUROC_00.pdf",
    "number_of_trees_Inductive_AUPRC_0_BICTR_Oxytrees.pdf": (
        "number_of_trees/Inductive_AUPRC_00_BICTR_Oxytrees.pdf"
    ),
    "number_of_trees_Inductive_AUROC_0_BICTR_Oxytrees.pdf": (
        "number_of_trees/Inductive_AUROC_00_BICTR_Oxytrees.pdf"
    ),
    "empirical_complexity_fit.pdf": (
        "empirical_complexity_plots/empirical_complexity_fit.pdf"
    ),
    "empirical_complexity_leaf_assign.pdf": (
        "empirical_complexity_plots/empirical_complexity_leaf_assign.pdf"
    ),
}


def main():
    for glob_pattern in GLOB_PATTERNS:
        for source in IN.rglob(glob_pattern):
            dest = OUT / source.relative_to(IN)
            dest.parent.mkdir(exist_ok=True, parents=True)

            print(f"Copying {source}")
            shutil.copy(source, dest)

    for dest_name, source_rel_path in CAMERA_READY_FIGURES.items():
        source = IN / source_rel_path
        dest = OUT_CAMERA_READY / dest_name
        dest.parent.mkdir(exist_ok=True, parents=True)

        print(f"Copying {source} to camera-ready as {dest_name}")
        shutil.copy(source, dest)

    print("Done.")



if __name__ == "__main__": 
    main()
