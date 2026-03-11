# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:52:56 2026

@author: KIOM_User
"""

# -*- coding: utf-8 -*-
"""
SwissTargetPrediction Data Curation Pipeline (Option A: Target-Centric + Year Info)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import StratifiedKFold
import os
import subprocess
import multiprocessing
from tqdm import tqdm

# ==========================================
# 1. 데이터베이스 연결 (수정됨: 연도 정보 추가)
# ==========================================
class ChemblDataFetcher:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def fetch_data(self):
        print(">> Fetching data from ChEMBL 36 database (including Year)...")
        
        # [수정] docs 테이블 조인 및 publication_year 추가
        query = """
        SELECT DISTINCT
            act.molregno,
            md.chembl_id AS mol_chembl_id,
            cs.canonical_smiles,
            act.standard_value,
            act.standard_type,
            act.standard_units,
            ass.confidence_score,
            d.year AS publication_year, -- [추가] 논문/특허 발행 연도
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_name,
            td.target_type,
            td.organism,
            c.accession AS uniprot_id,
            c.sequence
        FROM activities act
        JOIN assays ass ON act.assay_id = ass.assay_id
        LEFT JOIN docs d ON ass.doc_id = d.doc_id -- [추가] 문서 정보 조인
        JOIN target_dictionary td ON ass.tid = td.tid
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences c ON tc.component_id = c.component_id
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE
            ass.assay_type = 'B'
            AND td.organism IN ('Homo sapiens', 'Mus musculus', 'Rattus norvegicus')
            AND td.target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX')
            AND act.standard_type IN ('Ki', 'Kd', 'IC50', 'EC50')
            AND act.standard_value IS NOT NULL
            AND cs.canonical_smiles IS NOT NULL;
        """
        df = pd.read_sql(query, self.engine)
        print(f"   Fetched {len(df)} raw records.")
        return df

# ==========================================
# 2. 데이터 전처리
# ==========================================
def _calc_heavy_atoms(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol else 9999
    except:
        return 9999

def _calc_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        return None
    except:
        return None

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        max_limit = 60
        sys_cores = multiprocessing.cpu_count()
        self.n_cores = min(max_limit, sys_cores)

    def filter_heavy_atoms(self, max_atoms=80):
        print(f">> Filtering Heavy Atoms (Using {self.n_cores} cores)...")
        with multiprocessing.Pool(self.n_cores) as pool:
            self.df['heavy_atoms'] = pool.map(_calc_heavy_atoms, self.df['canonical_smiles'])
        self.df = self.df[self.df['heavy_atoms'] <= max_atoms].copy()
        print(f"   Remaining records: {len(self.df)}")

    def generate_scaffolds(self):
        print(f">> Generating Scaffolds (Using {self.n_cores} cores)...")
        with multiprocessing.Pool(self.n_cores) as pool:
            self.df['scaffold_smiles'] = pool.map(_calc_scaffold, self.df['canonical_smiles'])
        self.df.dropna(subset=['scaffold_smiles'], inplace=True)

    def standardize_activity(self):
        print(">> Standardizing Activity Values...")
        self.df['activity_uM'] = np.where(
            self.df['standard_units'] == 'nM',
            self.df['standard_value'] / 1000.0,
            self.df['standard_value']
        )
        conditions = [
            (self.df['activity_uM'] < 10),
            (self.df['activity_uM'] > 500)
        ]
        choices = ['Active', 'Inactive']
        self.df['label'] = np.select(conditions, choices, default='Ambiguous')

    def remove_ambiguity(self):
        print(">> Removing Ambiguous Data...")
        self.df = self.df[self.df['label'] != 'Ambiguous'].copy()
        grouped = self.df.groupby(['mol_chembl_id', 'target_chembl_id'])['label'].nunique()
        conflicting_pairs = grouped[grouped > 1].index
        self.df.set_index(['mol_chembl_id', 'target_chembl_id'], inplace=True)
        self.df.drop(index=conflicting_pairs, inplace=True)
        self.df.reset_index(inplace=True)
        # 중복 제거 시 가장 최신 연도의 데이터를 남기는 로직 등을 추가할 수 있으나, 
        # 여기서는 간단히 첫 번째 레코드를 유지
        self.df.drop_duplicates(subset=['mol_chembl_id', 'target_chembl_id'], inplace=True)
        print(f"   Cleaned records: {len(self.df)}")

    def add_chembl_version(self):
        self.df['source_version'] = 'ChEMBL_36_2025'

# ==========================================
# 3. BLAST (멀티스레드 적용)
# ==========================================
class BlastEngine:
    def __init__(self, df, work_dir="blast_work"):
        self.df = df
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        self.fasta_file = os.path.join(work_dir, "targets.fasta")
        self.db_name = os.path.join(work_dir, "target_db")
        self.paralog_map = {} 
        self.n_threads = str(multiprocessing.cpu_count()) 

    def prepare_blast_db(self):
        print(">> Preparing BLAST Database...")
        unique_targets = self.df[['target_chembl_id', 'sequence']].drop_duplicates().dropna()
        records = []
        for _, row in unique_targets.iterrows():
            records.append(SeqRecord(Seq(row['sequence']), id=row['target_chembl_id'], description=""))
        
        SeqIO.write(records, self.fasta_file, "fasta")
        
        cmd = ["makeblastdb", "-dbtype", "prot", "-in", self.fasta_file, "-out", self.db_name]
        subprocess.run(cmd, check=True, capture_output=True)

    def run_all_vs_all_blast(self, e_value_thresh=1e-5):
        print(f">> Running All-vs-All BLAST (Using {self.n_threads} threads)...")
        output_file = os.path.join(self.work_dir, "blast_results.xml")
        
        cmd = [
            "blastp",
            "-query", self.fasta_file,
            "-db", self.db_name,
            "-outfmt", "6",
            "-out", output_file,
            "-evalue", str(e_value_thresh),
            "-num_threads", self.n_threads
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        try:
            blast_df = pd.read_csv(output_file, sep='\t', header=None, 
                                   names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
            for _, row in blast_df.iterrows():
                if row['qseqid'] != row['sseqid']:
                    self.paralog_map.setdefault(row['qseqid'], set()).add(row['sseqid'])
        except pd.errors.EmptyDataError:
            print("   Warning: BLAST result is empty.")

# ==========================================
# 4. 데이터셋 빌더 (A안: 타겟 중심 + 연도 포함)
# ==========================================

_shared_all_mols = None
_shared_known_interactions = None
_shared_paralog_map = None
_shared_mol_meta = None

def _init_worker(all_mols, known_interactions, paralog_map, mol_meta):
    global _shared_all_mols, _shared_known_interactions, _shared_paralog_map, _shared_mol_meta
    _shared_all_mols = all_mols
    _shared_known_interactions = known_interactions
    _shared_paralog_map = paralog_map
    _shared_mol_meta = mol_meta

def _process_target_centric(pos_row_dict):
    """
    [Option A] Target Fixed, Random Ligand Sampling
    Paralog Filter Applied.
    Year info propagated.
    """
    all_mols = _shared_all_mols
    known_interactions_set = _shared_known_interactions
    paralog_map = _shared_paralog_map
    mol_meta = _shared_mol_meta
    
    query_mol = pos_row_dict['mol_chembl_id']
    query_target = pos_row_dict['target_chembl_id']
    
    # 1. Positive Entry (이미 publication_year 포함됨)
    pos_entry = pos_row_dict.copy()
    pos_entry['set_type'] = 'Positive'
    pos_entry['pair_id'] = f"{query_mol}_{query_target}"
    
    results = [pos_entry]
    
    # 2. Negative Sampling
    neg_samples = []
    selected_neg_mols = set()
    attempts = 0
    target_paralogs = paralog_map.get(query_target, set())
    
    while len(neg_samples) < 10 and attempts < 1000:
        attempts += 1
        rand_mol_id = np.random.choice(all_mols)
        
        if rand_mol_id == query_mol: continue
        if rand_mol_id in selected_neg_mols: continue
        if (rand_mol_id, query_target) in known_interactions_set: continue
        
        # Paralog Check
        is_paralog_active = False
        for p_target in target_paralogs:
            if (rand_mol_id, p_target) in known_interactions_set:
                is_paralog_active = True
                break
        if is_paralog_active: continue
        
        selected_neg_mols.add(rand_mol_id)
        
        # Negative Entry 생성
        neg_entry = pos_row_dict.copy() # 여기서 publication_year도 복사됨 (Positive의 연도 유지)
        neg_entry['mol_chembl_id'] = rand_mol_id
        
        if rand_mol_id in mol_meta:
            neg_entry['canonical_smiles'] = mol_meta[rand_mol_id]['canonical_smiles']
            neg_entry['heavy_atoms'] = mol_meta[rand_mol_id]['heavy_atoms']
            neg_entry['scaffold_smiles'] = mol_meta[rand_mol_id]['scaffold_smiles']
        
        neg_entry['activity_uM'] = np.nan
        neg_entry['standard_value'] = np.nan
        neg_entry['confidence_score'] = np.nan
        neg_entry['label'] = 'Assumed_Inactive'
        neg_entry['set_type'] = 'Negative'
        neg_entry['pair_id'] = f"{query_mol}_{query_target}"
        
        neg_samples.append(neg_entry)
        
    results.extend(neg_samples)
    return results

class DatasetBuilder:
    def __init__(self, df, blast_engine):
        self.df = df
        self.blast_engine = blast_engine
        self.final_dataset = []
        max_limit = 60
        sys_cores = multiprocessing.cpu_count()
        self.n_cores = min(max_limit, sys_cores)

    def build_balanced_dataset(self):
        print(f">> Building Datasets (Target-Centric / A-Type) with {self.n_cores} cores...")
        
        positives = self.df[self.df['label'] == 'Active']
        
        print("   Preparing lookup tables...")
        active_pairs = set(zip(self.df[self.df['label']=='Active']['mol_chembl_id'], 
                               self.df[self.df['label']=='Active']['target_chembl_id']))
        all_mols = self.df['mol_chembl_id'].unique()
        mol_meta = self.df[['mol_chembl_id', 'canonical_smiles', 'heavy_atoms', 'scaffold_smiles']] \
                       .drop_duplicates('mol_chembl_id') \
                       .set_index('mol_chembl_id') \
                       .to_dict('index')
        paralog_map = self.blast_engine.paralog_map
        tasks = positives.to_dict('records')
            
        print("   Starting multiprocessing pool...")
        processed_data = []
        with multiprocessing.Pool(processes=self.n_cores, 
                                  initializer=_init_worker, 
                                  initargs=(all_mols, active_pairs, paralog_map, mol_meta)) as pool:
            for result in tqdm(pool.imap_unordered(_process_target_centric, tasks, chunksize=1000), total=len(tasks)):
                processed_data.extend(result)
        
        self.final_dataset = pd.DataFrame(processed_data)
        print(f"   Total dataset size: {len(self.final_dataset)}")

    def add_cross_validation_folds(self, n_splits=10):
        print(">> Assigning CV folds...")
        unique_pairs = self.final_dataset[self.final_dataset['set_type'] == 'Positive']['pair_id'].unique()
        np.random.shuffle(unique_pairs)
        folds = np.array_split(unique_pairs, n_splits)
        fold_map = {}
        for fold_idx, pairs in enumerate(folds):
            for pair in pairs:
                fold_map[pair] = fold_idx + 1
        self.final_dataset['cv_fold'] = self.final_dataset['pair_id'].map(fold_map)

# ==========================================
# 메인 실행
# ==========================================
def main():
    # ★ DB 설정 수정 필요 ★
    DB_URL = "postgresql://postgres:99pqpeqt@localhost:5432/chembl_36"
    
    try:
        # 1. Fetch
        fetcher = ChemblDataFetcher(DB_URL)
        df_raw = fetcher.fetch_data()
        if df_raw.empty: return

        # 2. Preprocess
        preprocessor = DataPreprocessor(df_raw)
        preprocessor.filter_heavy_atoms()
        preprocessor.standardize_activity()
        preprocessor.remove_ambiguity()
        preprocessor.generate_scaffolds()
        preprocessor.add_chembl_version()
        df_clean = preprocessor.df
        
        # 3. BLAST
        blast_engine = BlastEngine(df_clean)
        blast_engine.prepare_blast_db()
        blast_engine.run_all_vs_all_blast()
        
        # 4. Dataset Build (A-Type + Year Info)
        builder = DatasetBuilder(df_clean, blast_engine)
        builder.build_balanced_dataset()
        builder.add_cross_validation_folds()
        
        # 5. Output
        final_df = builder.final_dataset
        
        # CSV 저장
        csv_file = "chembl36_stp_training_set_final_v2.csv"
        print(f">> Saving to CSV ({csv_file})...")
        final_df.to_csv(csv_file, index=False)
        
        # Parquet 저장
        parquet_file = "chembl36_stp_training_set_final_v2.parquet"
        print(f">> Saving to Parquet ({parquet_file})...")
        try:
            final_df.to_parquet(parquet_file, index=False, compression='snappy')
            print("   Parquet save success.")
        except ImportError:
            print("   [Warning] Installing 'pyarrow' for Parquet support...")
            subprocess.check_call(["pip", "install", "pyarrow"])
            final_df.to_parquet(parquet_file, index=False, compression='snappy')
            print("   Parquet save success.")

        print(">> All processes completed successfully.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()