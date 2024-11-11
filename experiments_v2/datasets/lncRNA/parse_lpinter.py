from argparse import ArgumentParser
import pandas as pd
from pathlib import Path


def parse_lpinter(data_path, interactions_path, lncrna_path, targets_path):
    data = pd.read_csv(data_path, sep='\t', dtype=str)

    # Select only lncRNA-protein interactions
    data = data[
        (data.ncType == 'lncRNA')
        & (data.tarType == 'protein')
        & (data['class'] == 'binding')  # Exclude coexpression data
        & (data.level == 'RNA-Protein')
        & (data.organism == 'Homo sapiens')  # See below
    ]

    # ===== Number of interactions per species ======
    # Homo sapiens                             391607
    # Mus musculus                              94935
    # Saccharomyces cerevisiae                    412
    # Agrobacterium tumefaciens                   208
    # Caenorhabditis elegans                       30
    # Drosophila melanogaster                      27
    # Escherichia coli                             26
    # Other                                        35

    # Select only the first ID (multiple IDs are separated by ';')
    data['tarID'] = data['tarID'].str.split(';', n=1).str[0]
    data[['ncID', 'tarID']].to_csv(interactions_path, sep='\t', index=False)
    print(f'Interactions saved to {interactions_path}.')

    target_ids = pd.Series(data.tarID.unique())
    lncrna_ids = pd.Series(data.ncID.unique())

    target_ids.to_csv(
        targets_path,
        index=False,
        header=False,
    )
    print(f'UniProt IDs for target proteins saved to {targets_path}.')

    lncrna_ids.to_csv(
        lncrna_path,
        index=False,
        header=False,
    )
    print(f'NONCODE IDs for lncRNAs saved to {lncrna_path}.')


def main():
    parser = ArgumentParser(
        description=(
            'Parse NPInter data file to extract lncRNA-protein interactions.'
            '\nThree text files are generated:'
            '\n  - lncRNA accession codes for the NONCODE database'
            '\n  - UniProt accession codes of the target proteins'
            '\n  - Two-column TSV file listing lncRNA-protein IDs for each interaction'
            '\nThe NPInter dataset file (--data-path argument) can be downloaded from'
            'http://bigdata.ibp.ac.cn/npinter4/download/.'
        )
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        help='Path to the NPInter data file.',
        required=True,
    )
    parser.add_argument(
        '--interactions-path',
        type=Path,
        help='Path to the output file with lncRNA-protein interactions.',
        required=True,
    )
    parser.add_argument(
        '--lncrna-path',
        type=Path,
        help='Path to the output file with lncRNA NONCODE IDs.',
        required=True,
    )
    parser.add_argument(
        '--targets-path',
        type=Path,
        help='Path to the output file with target protein UniProt IDs.',
        required=True,
    )
    args = parser.parse_args()

    # TODO: fetch URL
    # http://bigdata.ibp.ac.cn/npinter4/download/file/interaction_NPInterv4.txt.gz
    # data_path = Path('interaction_NPInterv4.txt.gz')
    # interactions_path = data_path.with_name('lncrna_protein_interactions.tsv')
    # lncrna_path = data_path.with_name('lncrna_noncode_ids.txt')
    # targets_path = data_path.with_name('protein_target_uniprot_ids.txt')

    parse_lpinter(
        data_path=args.data_path,
        interactions_path=args.interactions_path,
        lncrna_path=args.lncrna_path,
        targets_path=args.targets_path,
    )



if __name__ == '__main__':
    main()