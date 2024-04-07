import random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
import pickle
from tqdm import tqdm
import numpy as np
from rdkit import Chem


def train_test_split(mol_list, seed=2,test_proportion=0.2, split_scaffold=False):
    """
    mol_list: molecule list
    seed: seed number
    test_proportion: proportion for the test data
    split_scaffold: whether split scaffold or not(true/false)
    """
    random.seed(seed)
    scaffolds_sets = defaultdict(list)

    for idx, mol in enumerate(tqdm(mol_list)):
        if split_scaffold:
        
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
            scaffolds_sets[scaffold].append(idx)
        else:
            smi = Chem.MolToSmiles(mol)
            scaffolds_sets[smi].append(idx)
    
    test_data_size = test_proportion*len(mol_list)
    train,test = list(),list()
    index_sets = list(scaffolds_sets.values())
    random.shuffle(index_sets)
    for index_set in index_sets:
        if len(test)+len(index_set)<=test_data_size:
            test += index_set
        else:
            train += index_set
            
    return train,test

