from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import cairosvg


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def smiles2png(smiles, filename):
    m = Chem.MolFromSmiles(smiles)
    AllChem.ComputeGasteigerCharges(m)
    for at in m.GetAtoms():
        lbl = '%.2f'%(at.GetDoubleProp("_GasteigerCharge"))
        at.SetProp('atomNote',lbl)
    m = mol_with_atom_index(m)
    img = Draw.MolToFile(m, 'temp.svg')

    input_svg = 'temp.svg'
    output_png = filename + '.png'
    dpi = 300
    output_width = 800  # 输出图片宽度
    output_height = 800  # 输出图片高度
    cairosvg.svg2png(
        url=input_svg, write_to=output_png,
        dpi=dpi, output_width=output_width, 
        output_height=output_height)
    return