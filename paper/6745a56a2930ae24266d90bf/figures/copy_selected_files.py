from pathlib import Path
import shutil


IN = Path("../../../experiments_v2/results").resolve()
OUT = Path("./results").resolve()
GLOB_PATTERNS = (
    "*/*/all_datasets/critical_difference_diagrams/*.pdf",
    "latex_tables/0/*.tex",
    "latex_tables/0/*.pdf",
    "empirical_complexity/*.pdf",
    "*/*/parallel_coordinates/*.pdf",
)


def main():
    for glob_pattern in GLOB_PATTERNS:
        for source in IN.rglob(glob_pattern):
            dest = OUT / source.relative_to(IN)
            dest.parent.mkdir(exist_ok=True, parents=True)

            print(f"Copying {source}")
            shutil.copy(source, dest)

    print("Done.")



if __name__ == "__main__": 
    main()
