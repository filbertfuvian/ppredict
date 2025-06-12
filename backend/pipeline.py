import pandas as pd
from Bio import SeqIO
import requests
import time

def read_scop_classification(file_path):
    cols = [
        'FA_DOMID', 'FA_PDBID', 'FA_PDBREG', 'FA_UNIID', 'FA_UNIREG',
        'SF_DOMID', 'SF_PDBID', 'SF_PDBREG', 'SF_UNIID', 'SF_UNIREG', 'SCOPCLA'
    ]
    df = pd.read_csv(file_path)
    return df

def read_fasta_sequences(fasta_path):
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_dict[record.id] = str(record.seq)
    return seq_dict

def is_homo_sapiens(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url)
        print(f"Trying for {uniprot_id}...")
        if response.status_code == 200:
            data = response.json()
            organism = data.get('organism', {})
            if organism.get('scientificName', '').lower() == 'homo sapiens':
                return True
        else:
            print(f"UniProt ID {uniprot_id} not found (status {response.status_code})")
    except Exception as e:
        print(f"Error fetching UniProt data for {uniprot_id}: {e}")
    return False


def process_scop_data(scop_class_file, fasta_file, output_csv=None):
    df = read_scop_classification(scop_class_file)
    print(f"Total records in SCOP classification: {len(df)}")

    unique_uniprot_ids = df['FA_UNIID'].unique()
    print(f"Unique UniProt IDs: {len(unique_uniprot_ids)}")

    homo_sapiens_ids = []
    for uid in unique_uniprot_ids:
        if is_homo_sapiens(uid):
            homo_sapiens_ids.append(uid)

    print(f"Homo sapiens UniProt IDs found: {len(homo_sapiens_ids)}")

    df_hs = df[df['FA_UNIID'].isin(homo_sapiens_ids)]


    seq_dict = read_fasta_sequences(fasta_file)

    df_hs = df_hs.copy() 
    df_hs['Sequence'] = df_hs['FA_DOMID'].astype(str).map(seq_dict)

    df_hs = df_hs.dropna(subset=['Sequence'])

    print(f"Final dataset size after filtering and merging: {len(df_hs)}")

    if output_csv:
        df_hs.to_csv(output_csv, index=False)
        print(f"Result saved to {output_csv}")

    return df_hs

if __name__ == "__main__":
    scop_class_file = "out_scop.csv"
    fasta_file = "scop_fa_represeq_lib_latest.fa"
    output_csv = "scop_hs_top50.csv"
    df_final = process_scop_data(scop_class_file, fasta_file, output_csv=output_csv)
    print(df_final.head())
