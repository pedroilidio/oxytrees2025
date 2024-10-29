# Steps to generate the bipartite dataset

1. The inteaction data is recovered from NPInter, at the following URL:
    * [http://bigdata.ibp.ac.cn/npinter4/download/file/interaction_NPInterv4.txt.gz]

2. Run parse_lpinter.py to extract main information: 
    ```bash
    $ python parse_lpinter.py \
    --data_path=interaction_NPInterv4.txt.gz \
    --interactions_path=lncrna_protein_interactions.tsv \
    --lncrna_path=lncrna_noncode_ids.txt \
    --targets_path=protein_target_uniprot_ids.txt
    ```
    * Outputs:
        * lncrna_protein_interactions.tsv
        * lncrna_noncode_ids.txt
        * protein_target_uniprot_ids.txt

3. NONCODE data with the lncRNA sequences for *Homo sapiens* must be retrieved from the following URL:
    * [http://www.noncode.org/datadownload/NONCODEv6_human.fa.gz]
    (which is available at [http://www.noncode.org/download.php])
    * Output: NONCODEv6_human.fa

4. Protein sequences are recovered from UniProt by inputting the IDs file (protein_target_uniprot_ids.txt) to its batch search interface (https://www.uniprot.org/id-mapping), downloading and uncompressing the fasta file resulting from the query.
    * Output: uniprot-compressed_true_download_true_format_fasta-XXXX.XX.XX-XX.XX.XX.XX.fasta
    
5. Run make_interaction_dataset.py to generate the interaction matrix and similarity matrices from previous files:
    ```bash
    $ python make_interaction_dataset.py \
    --interactions_table_input=lncrna_protein_interactions.tsv \
    --lncrna_fasta_input=NONCODEv6_human.fa \
    --target_fasta_input=uniprot-compressed_true_download_true_format_fasta-XXXX.XX.XX-XX.XX.XX.XX.fasta
    --found_interactions_output=found_interactions.tsv \
    --lncrna_output=lncrna_seqs.tsv \
    --target_output=target_seqs.tsv \
    --lncrna_similarity_output=lncrna_similarity.tsv \
    --target_similarity_output=target_similarity.tsv \
    --norm_target_output=normalized_target_similarity.tsv \
    --norm_lncrna_similarity_output=normalized_lncrna_similarity.tsv \
    --interaction_matrix_output=interaction_matrix.tsv
    --n_jobs=1  # Adjust to your needs
    ```
    * Outputs:
        * found_interactions.tsv
        * lncrna_seqs.tsv
        * target_seqs.tsv
        * lncrna_similarity.tsv
        * target_similarity.tsv
        * normalized_target_similarity.tsv
        * normalized_lncrna_similarity.tsv
        * interaction_matrix.tsv

Note: further details of each step are presented in the --help of each corresponding script.