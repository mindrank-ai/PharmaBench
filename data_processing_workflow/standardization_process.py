import pandas as pd
import numpy as np
from rdkit import Chem
from scipy.stats import pearsonr
from Draw_fig import Draw_plot,draw_confusion
import random
random.seed(68)
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score
from rdkit.Chem.Scaffolds import MurckoScaffold


'''
Step_1
1. Smiles处理
function: smiles_check -> 应用rdkit的Chem.SanitizeMol()函数检查化合物SMILES正确性
function: remove_salt -> 对盐/溶剂的处理
function: metal_check -> 对重金属的处理
function: smiles_unify -> 统一SMILES格式

'''
def smi_to_smi(x):
    try:
        mol = Chem.MolFromSmiles(x)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        return Chem.MolToSmiles(mol)
    except:
        return np.nan

def smiles_check(df, smi_col):
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    returns:
        df: DataFrame with a new column named 'Smiles_check'
        print(df['Smiles_check'].value_counts(dropna=False)):
        if all values are SANITIZE_NONE means all SMILES are right, if False come out means corresponding SMILES are wrong which need furthur amend
    '''
    mols = [Chem.MolFromSmiles(x) for x in df[smi_col].values]
    check_result = []
    for mol in tqdm(mols):
        a = bool(mol)
        if a:
            b = str(Chem.SanitizeMol(mol))
            check_result.append(b)
        else:
            check_result.append(a)

    df['Smiles_check'] = check_result
    print(df['Smiles_check'].value_counts(dropna=False))
    return df


def remove_salt(df, smi_col):
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    returns:
        df: DataFrame with a new column named 'Smiles_removesalt': a col contains SMILES without any salt/solvent
    '''
    Smiles_rs = []
    for smiles in tqdm(df[smi_col].values):
        frags = smiles.split(".")
        frags = sorted(frags, key=lambda x: len(x), reverse=True)
        Smiles_rs.append(frags[0])

    df['Smiles_removesalt'] = Smiles_rs
    return df


METALS = (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32,33,
          37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,52,55, 56, 57,
          58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
          75, 76, 77, 78, 79, 80, 81, 82, 83, 84,85,87, 88, 89, 90, 91, 92, 93, 94,
          95, 96, 97, 98, 99, 100, 101, 102, 103) #3, 4, 11, 12, 13, 19, 20,
def is_transition_metal(atom, METALS):
    n = atom.GetAtomicNum()
    return (n in METALS)

def metal_check(df, smi_col):
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    returns:
        Metal_check_idx: a list contain df.index which corresponding Smiles has Metal
    '''
    mols = [Chem.MolFromSmiles(x) for x in df[smi_col].values]
    Metal_check_idx = []
    for idx, mol in tqdm(zip(df.index, mols)):
        for atom in mol.GetAtoms():
            if (is_transition_metal(atom, METALS)):
                Metal_check_idx.append(idx)
                break
    print('Metal in Smiles amount:', len(Metal_check_idx))
    return Metal_check_idx


def smiles_unify(df, smi_col):
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    returns:
        df: DataFrame with a new column named 'Smiles_unify': a col contains unified SMILES
    '''
    mols = [Chem.MolFromSmiles(x) for x in df[smi_col].values]
    Smiles_unify = []
    for m in tqdm(mols):
        s_u = Chem.MolToSmiles(m)
        Smiles_unify.append(s_u)

    df['Smiles_unify'] = Smiles_unify
    return df




'''
Step_2
1. 限定非氢原子数目 > 10
function: AtomCounts -> 限定非氢原子数目

'''

def AtomCounts(df, smi_col, count=10):
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
        count: The number of non-hydrogens defined is greater than, float
    returns:
        df_atomcount: new DataFrame which has all compounds non-hydrogen atoms > count
    '''

    df['AtomCounts'] = df[smi_col].apply(lambda x: Chem.MolFromSmiles(x).GetNumAtoms())
    df_atomcount = df[df['AtomCounts'] > count]
    df_atomcount = df_atomcount[df_atomcount['AtomCounts'] <=100]

    print('alldata shape:', df.shape)
    print('AtomCounts shape:', df[df['AtomCounts'] > count].shape)
    return df_atomcount




'''
Step_3
1. 各来源数据集之间的相关性评估
function: plot_reg_correlation -> 回归数据集之间的相关性评估
function: plot_cls_correlation -> 分类数据集之间的相关性评估

'''

def reg_coef(x,y,label = None, color = None, **kwargs):
    ax = plt.gca()
    ax.figure.set_size_inches(8, 8)
#     print(y)
    nas = np.logical_or(x.isna(), y.isna())
#     print(len(x[~nas]),len(y[~nas]))
    if (len(x[~nas])<=2):
        r = 0
        p = 'No matched'
    else:
        r,p  = pearsonr(x[~nas], y[~nas])
        if p < 0.01:
            p=(' < 0.01')
        else:
            p=(str(round(p,3)))

    ax.annotate('Corr :{:.3f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
    ax.annotate('P_value :{}'.format(p), xy=(0.5,0.4), xycoords='axes fraction', ha='center')
#     ax.annotate('P_value :{:.3f}'.format(p), xy=(0.5,0.4), xycoords='axes fraction', ha='center')
    ax.set_axis_off()

def plot_reg_correlation(df, smi_cols, source_col, value_col, fig_title):
    '''
    Args:
        df: DataFrame
        smi_cols: a list contains (SMILES or other controlled conditions') column name, list
        source_col: data_Source column name, str
        value_col: value column name, str
        fig_title: customized reg_correlation figure' title, str
    returns:
        reg_correlation figure
    '''
    def draw_reg_plot(x,y,**kwargs):
        plt.plot([data_min, data_max], [data_min, data_max], color='grey', linestyle='solid')
        sns.regplot(x=x,y=y, scatter_kws={'s': 5})

    df_pivot = pd.pivot_table(data=df, index=smi_cols, columns=source_col, values=value_col)
    df_pivot.reset_index(inplace=True)

    df_data = df_pivot.iloc[:, len(smi_cols):] # df_data只包含求相关性系数值的列

    data_max = int(np.ceil(df_data.max().max()))
    data_min = int(np.floor(df_data.min().min()))
    pairs = sns.PairGrid(df_data)
    pairs.map_diag(sns.histplot) #diagonal
    pairs.map_lower(draw_reg_plot)
    pairs.map_upper(reg_coef) #Upper
    pairs.fig.suptitle(fig_title, y=1.08)

    data_range = np.linspace(data_min,data_max,7)

    for ax in pairs.axes.flat:
        ax.set_xticks(data_range)
        ax.set_yticks(data_range)


# def plot_cls_correlation(df, smi_cols, source_col, value_col, fig_title):
#     '''
#     Args:
#         df: DataFrame
#         smi_cols: a list contains (SMILES or other controlled conditions') column name, list
#         source_col: data_Source column name, str
#         value_col: value column name, str
#         fig_title: customized reg_correlation figure' title, str
#     returns:
#         reg_correlation figure
#     '''
#     df_pivot = pd.pivot_table(data=df, index=smi_cols, columns=source_col, values=value_col)
#     df_pivot.reset_index(inplace=True)

#     df_data = df_pivot.iloc[:, len(smi_cols):] # df_data只包含求相关性系数值的列

#     sns.heatmap(df_data.corr(), annot=True, vmin=0, vmax=1, square=True)


def plot_cls_correlation(df, smi_cols, source_col, value_col, fig_title):
    '''
    Args:
        df: DataFrame
        smi_cols: a list contains (SMILES or other controlled conditions') column name, list
        source_col: data_Source column name, str
        value_col: value column name, str
        fig_title: customized reg_correlation figure' title, str
    returns:
        reg_correlation figure
    '''
    df  = df.drop_duplicates(smi_cols + [source_col])
    df_pivot = pd.pivot_table(data=df, index=smi_cols, columns=source_col, values=value_col)
    df_pivot.reset_index(inplace=True)

    df_data = df_pivot.iloc[:, len(smi_cols):] # df_data只包含求相关性系数值的列
    col_list = df_data.columns
    result = defaultdict(list)
    for i in range(len(col_list)):
        for j in range(len(col_list)):
            col1 = col_list[i]
            col2 = col_list[j]
            tmp = df_data[(~df_data[col1].isna()) & (~df_data[col2].isna())]
            if tmp.empty:
                result[col1].append(np.nan)
                continue
            acc = accuracy_score(tmp[col1], tmp[col2])
            result[col1].append(acc)
    result_df = pd.DataFrame(result, index=col_list)
    sns.heatmap(result_df, annot=True, vmin=0, vmax=1, square=True)



'''
Step_3
2. 对相同SMILES重复值的处理
    2. function: plot_diagram -> 查看相同SMILES数据重复性
    3. function: unreliable_datapoints -> 挑出相同SMILES重复值中可信度较低的数据
    4. function: identify_unreliable_group -> 挑出相同SMILES重复值里（最大值/最小值 > fold -> 根据性质决定，如LogD建议fold=log(3)）的SMILES所有数据
    5. function: final_mean_value -> 剩余相同SMILES重复值取平均值

'''

def plot_diagram(df, smi_cols, value_col):
    '''
    Args:
        df: DataFrame
        smi_cols: a list contains (SMILES or other controlled conditions') column name, list
                    (SMILES or other controlled conditions') column better fillna, which means no NA values
        value_col: value column name, str
    returns:
        a figure presents data producibility
    '''

    result_list = []

    # Identify duplicated df and processed the data within the duplicated df
    # Only process the df with duplicated values
    duplicated_df = df[df.duplicated(subset=smi_cols,keep=False)]
    for group in duplicated_df.groupby(smi_cols):

        value_list = [x for x in group[1][value_col]]
        result = [min(value_list),max(value_list)]
        random.shuffle(result)
        result_list += [result]

    value_list_1 = [x[0] for x in result_list]
    value_list_2 = [x[1] for x in result_list]
    max_list_1,min_list_1 = max(value_list_1), min(value_list_1)
    max_list_2,min_list_2 = max(value_list_2), min(value_list_2)
    Draw_plot(value_list_1, value_list_2,
          Fig_title = '',
          x_title = f'Experimental 1',
          y_title = f'Experimental 2',
          x_axis_mv=(max(max_list_1+2,max_list_2+2)-min(min_list_1-2,min_list_2-2))+2,
        y_axis_mv=(max(max_list_1+2,max_list_2+2)-min(min_list_1-2,min_list_2-2))/10,
        fold4 = 0.5 ,
        fold8 = 1,
        xmin=min(min_list_1-2,min_list_2-2),
        xmax=max(max_list_1+2,max_list_2+2),
        ymin=min(min_list_1-2,min_list_2-2),
        ymax =max(max_list_1+2,max_list_2+2))

def plot_classification_diagram(df, smi_cols, value_col):
    '''
    Args:
        df: DataFrame
        smi_cols: a list contains (SMILES or other controlled conditions') column name, list
                    (SMILES or other controlled conditions') column better fillna, which means no NA values
        value_col: value column name, str
    returns:
        a figure presents data producibility
    '''

    result_list = []

    # Identify duplicated df and processed the data within the duplicated df
    # Only process the df with duplicated values
    duplicated_df = df[df.duplicated(subset=smi_cols,keep=False)]
    for group in duplicated_df.groupby(smi_cols):

        value_list = [x for x in group[1][value_col]]
        result = [min(value_list),max(value_list)]
        random.shuffle(result)
        result_list += [result]

    value_list_1 = [x[0] for x in result_list]
    value_list_2 = [x[1] for x in result_list]
    max_list_1,min_list_1 = max(value_list_1), min(value_list_1)
    max_list_2,min_list_2 = max(value_list_2), min(value_list_2)
    draw_confusion(value_list_1, value_list_2,['0','1'])


def unreliable_datapoints(df, smi_cols, value_col):
    '''
    Args:
        df: DataFrame
        smi_cols: a list contains (SMILES or other controlled conditions') column name, list
                    (SMILES or other controlled conditions') column better fillna, which means no NA values
        value_col: value column name, str
    returns:
        bad_index: a list contains unreliable datapoints' index
    '''

    bad_index = []

    for group in df.groupby(smi_cols):
        if len(group[1]) >= 3:
            value_list = [x for x in group[1][value_col]]
            max_value = np.mean(value_list) + np.std(value_list)
            min_value = np.mean(value_list) - np.std(value_list)

            tmp = group[1]
            tmp = tmp[(tmp[value_col]<min_value)|(tmp[value_col]>max_value)]
            bad_index += list(tmp.index)

    return bad_index


def identify_unreliable_group(df, smi_cols, value_col, fold=np.log10(3)):
    '''
    Args:
        df: DataFrame
        smi_cols: a list contains (SMILES or other controlled conditions') column name, list
                    (SMILES or other controlled conditions') column better fillna, which means no NA values
        value_col: value(log) column name, str
        fold: number of error, float
    returns:
        bad_group_index: a list contains unreliable datapoints' index
    '''

    bad_group_index = []

    # Identify duplicated df and processed the data within the duplicated df
    # Only process the df with duplicated values
    duplicated_df = df[df.duplicated(subset=smi_cols,keep=False)]

    for group in duplicated_df.groupby(smi_cols):

        value_list = [x for x in group[1][value_col]]
        if max(value_list) - min(value_list) > fold:
            bad_group_index += list(group[1].index)

    return bad_group_index


def final_mean_value(df, smi_cols, value_col):
    '''
    Args:
        df: DataFrame
        smi_cols: a list contains (SMILES or other controlled conditions') column name, list
                    (SMILES or other controlled conditions') column better fillna, which means no NA values
        value_col: value column name, str
    returns:
        df: DataFrame with a new column named f"{value_col}_mean" containing mean value
    '''

    groups = []
    for group in df.groupby(smi_cols):
        value_list = [x for x in group[1][value_col]]
        mean_v = np.mean(value_list)

        df.loc[group[1].index, f"{value_col}_mean"] = mean_v

    return df


def process_stage_1(df,smi_col):
    df[smi_col] = df[smi_col].parallel_apply(smi_to_smi)
    df = df[~df[smi_col].isna()]
    df = smiles_check(df , smi_col)
    df = df[df['Smiles_check'] != False]
    df = remove_salt(df, smi_col)
    metal_index = metal_check(df, 'Smiles_removesalt')
    df = df[~df.index.isin(metal_index)]
    df = smiles_unify(df, 'Smiles_removesalt')
    return df 
