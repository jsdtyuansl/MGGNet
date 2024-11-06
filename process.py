import os
import pickle
import shutil
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def generate_pocket_file(data_dir, cut=5):
    print(f'---generate_pocket within {cut}A---')
    complex_id = os.listdir(data_dir)  # PDBid组成的列表
    for code in complex_id:
        complex_dir = os.path.join(data_dir, code)
        lig_path = os.path.join(complex_dir, f"{code}_ligand.mol2")
        protein_file = os.path.join(complex_dir, f"{code}_protein.pdb")

        if os.path.exists(os.path.join(complex_dir, f'Pocket_{cut}A.pdb')):
            continue

        pymol.cmd.load(protein_file)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket',
                         f'byres {code}_ligand around {cut}')  # 选择'Pocket'区域，该区域包括与指定的配体（{cid}_ligand）在一定距离范围内的残基  这边应该原子会多一点！！！
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{cut}A.pdb'),
                       'Pocket')
        pymol.cmd.delete('all')


def generate_file(data_dir, data_df, cut=5, ligand_format='mol2'):
    generate_pocket_file(data_dir=data_dir, cut=cut)
    count = 0
    for i, row in tqdm(data_df.iterrows()):
        code, pKa = row['PDB_code'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, code)  # ./data/set/+code
        pocket_file = os.path.join(data_dir, code, f'Pocket_{cut}A.pdb')
        # mol2文件
        ligand_input_path = os.path.join(data_dir, code, f'{code}_ligand.{ligand_format}')
        ligand_file = ligand_input_path.replace(f".{ligand_format}", ".pdb")
        os.system(f'obabel {ligand_input_path} -O {ligand_file} -d')  # -d 删除氢

        # 生成图路径
        save_path = os.path.join(complex_dir, f"{code}_{cut}A.pkl")
        ligand = Chem.MolFromPDBFile(ligand_file)
        pocket = Chem.MolFromPDBFile(pocket_file)

        if ligand is None or pocket is None:
            print(f"process {code} error")
            count += 1
            # 删除rdkit无法处理的mol
            # shutil.rmtree(os.path.join(data_dir, code))
            continue

        # 得到rdkit要处理的mol文件
        mol_file = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(mol_file, f)
    print("-----", count)


if __name__ == '__main__':
    # total有99无法处理
    cut = 5
    data_root = './data'

    for set_name in ['draw_set']:
        data_dir = os.path.join(data_root, set_name)
        data_df = pd.read_csv(os.path.join(data_root, f'{set_name}.csv'))
        generate_file(data_dir, data_df, cut=cut)
