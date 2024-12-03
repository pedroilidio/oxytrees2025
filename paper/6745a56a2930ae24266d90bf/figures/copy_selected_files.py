from pathlib import Path
import shutil


IN = Path("../../../experiments_v2/results").resolve()
OUT = Path("./results").resolve()
GLOB_PATTERN = "*/*/all_datasets/boxplots/*.pdf"


def main():
    for source in IN.rglob(GLOB_PATTERN):
        dest = OUT / source.relative_to(IN)
        dest.parent.mkdir(exist_ok=True, parents=True)

        print(f"Copying {source}")
        shutil.copy(source, dest)

    print("Done.")



if __name__ == "__main__": 
    main()
