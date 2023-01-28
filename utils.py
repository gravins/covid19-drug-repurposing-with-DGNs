import tqdm
import dill
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


def read_gene2drug_files():
    drug_host_uniprot = pd.read_csv('dataset/raw/drug-host_uniprot.tab', sep='\t')
    drug_host = pd.read_csv('dataset/raw/drug-host.tab', sep='\t')

    gene2drug = dict()
    for _, row in drug_host.iterrows():
        geneid, drug = str(row['EntrezGeneID']), str(row['DrugBankID'])
        if geneid not in gene2drug: gene2drug[geneid] = set()
        gene2drug[geneid].add(drug)

    for _, row in drug_host_uniprot.iterrows():
        geneid = str(row['EntrezGeneID'])
        if ',' in geneid: geneid = geneid.split(',')
        if pd.isnull(row['DrugBankIDs']): continue
        drugss = str(row['DrugBankIDs'])[:-1].split(';')
        if type(geneid) == list:
            for gene in geneid:
                for drug in drugss:
                    if gene not in gene2drug: gene2drug[gene] = set()
                    gene2drug[gene].add(drug)
        else:
            for drug in drugss:
                if geneid not in gene2drug: gene2drug[geneid] = set()
                gene2drug[geneid].add(drug)
    return gene2drug


def drug2genes():
    drug_host_uniprot = pd.read_csv('dataset/raw/drug-host_uniprot.tab', sep='\t')
    drug_host = pd.read_csv('dataset/raw/drug-host.tab', sep='\t')

    d2g = dict()
    for _, row in drug_host.iterrows():
        geneid, drug = str(row['EntrezGeneID']), str(row['DrugBankID'])
        drug = drug.strip()
        geneid = geneid.strip()
        if drug not in d2g: d2g[drug] = set()
        d2g[drug].add(geneid)

    for _, row in drug_host_uniprot.iterrows():
        geneid = str(row['EntrezGeneID'])
        if ',' in geneid: geneid = geneid.split(',')
        if pd.isnull(row['DrugBankIDs']): continue
        drugs = str(row['DrugBankIDs'])[:-1].split(';')
        for d in drugs:
            d = d.strip()
            if d not in d2g: d2g[d] = set()
            if type(geneid) == list:
                d2g[d] |= set([g.strip() for g in geneid])
            else:
                d2g[d].add(geneid.strip())
    return d2g


def draw_plot(tr_val, valid_val=None, y_label="", title="", vline=False, log=True, mode=None, path="./plot.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    if valid_val is not None:
        ax.plot(tr_val, label="Training")

        lab = mode if mode else "Validation"
        ax.plot(valid_val, label=lab)
    else:
        ax.plot(tr_val)

    if vline:
        if valid_val is not None:
            i = valid_val.index(max(valid_val))
            label = "i:" + str(i) + "  tr:" + str(round(tr_val[i],4)) + '  eval:' + str(round(valid_val[i], 4))
            ax.axvline(i, linestyle="--", color="#FFD700")
            plt.text(i, min(valid_val) + 0.05,  s=label, rotation=90, verticalalignment='center')
        else:
            i = tr_val.index(max(tr_val))
            label = "i:" + str(i) + "  eval:" + str(round(tr_val[i],4))
            ax.axvline(i, linestyle="--", color="#FFD700")
            plt.text(i, min(tr_val) + 0.05, s=label, rotation=90, verticalalignment='center')

    if log:
        ax.set_yscale('log')
    ax.set_ylabel(y_label)
    ax.set_xlabel("Epochs")
    ax.set_title(title)

    if valid_val is not None:
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def normal(m):
    m = np.asarray(m)
    row_sums = m.sum(axis=1)
    m = m / row_sums[:, np.newaxis]
    return m


def plot_confusion_matrix(conf, name, path="cm.png", norm=True, xlab=("0", "1")):
    # Set up the matplotlib figure
    f, ax = plt.subplots()

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    if norm:
        # Normalize confusion matrix
        conf = normal(conf)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(conf, cmap=cmap, center=0.5, vmax=1, ax=ax, xticklabels=xlab, yticklabels=xlab,
                annot=True, square=False, linewidths=.1, cbar=True, cbar_kws={"shrink": .87})

    plt.yticks(rotation=0)
    plt.title(name)
    plt.tight_layout()
    f.savefig(path, dpi=400)
    plt.close()


def make_plots(history, title, name, mode):
    scores = {
              "loss": {"tr": [], "val": []},
              "f1": {"tr": [], "val": []},
              "roc_auc": {"tr": [], "val": []}
            }

    for h in history:
        scores['loss']['tr'].append(h['Training']['loss'])
        scores['f1']['tr'].append(h['Training']['f1_score'])
        scores['roc_auc']['tr'].append(h['Training']['roc_auc'])
        scores['loss']['val'].append(h[mode]['loss'])
        scores['f1']['val'].append(h[mode]['f1_score'])
        scores['roc_auc']['val'].append(h[mode]['roc_auc'])
    
    # Plot the loss    
    draw_plot(scores["loss"]["tr"], scores["loss"]["val"], y_label="Loss", title=title, log=False, mode=mode, path="loss_" + name + ".png")
    draw_plot(scores["loss"]["tr"], scores["loss"]["val"], y_label="Loss", title=title, log=True, mode=mode, path="loss_" + name + "_log_scale.png")
    
    # Plot f1
    draw_plot(scores["f1"]["tr"], scores["f1"]["val"], y_label="F1 score", title=title, vline=True, log=False, mode=mode, path="f1_" + name + ".png")
    draw_plot(scores["f1"]["tr"], scores["f1"]["val"], y_label="F1 score", title=title, vline=True, log=True, mode=mode, path="f1_" + name + "_log_scale.png")
    
    # Plot roc auc
    draw_plot(scores["roc_auc"]["tr"], scores["roc_auc"]["val"], y_label="AUROC score", title=title, vline=True, log=False, mode=mode, path="auroc_" + name + ".png")
    draw_plot(scores["roc_auc"]["tr"], scores["roc_auc"]["val"], y_label="AUROC score", title=title, vline=True, log=True, mode=mode, path="auroc_" + name + "_log_scale.png")

    # Plot confusion matrix
    #cm = normal(history[scores['roc_auc']['val'].index(max(scores['roc_auc']['val']))][mode + " confusion_matrix"])
    #plot_confusion_matrix(cm, title, "cm_" + name + ".png")


