from torch_geometric.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os


def get_output(dataset, model, cuda, cpu, batchsize):
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
    booking = dataset.booking
    model.eval()
    outputs = []
    weights = []
    labels = []
    for data in loader:
        data.to(cuda)
        output = (
            model(data.x, data.edge_index, data.edge_attr, data.u, data.pseudo_mH, data.batch)
            .detach()
            .to(cpu)
            .numpy()
            .reshape(-1, 1)
        )
        outputs.append(output)
        weights.append(data.w.to(cpu).numpy().reshape(-1, 1))
        labels.append(data.y.to(cpu).numpy().reshape(-1, 1))
    return np.concatenate(outputs, axis=0), np.concatenate(labels, axis=0), np.concatenate(weights, axis=0), booking


# Get score predictions and construct dataframe of results
def get_df_results(bookings, outputs, labels, weights):
    df = pd.DataFrame()
    df["nJets"] = bookings["nJets"]
    df["nBTags"] = bookings["nBTags_DL1r_70"]
    df["mass"] = bookings["mH_label"]
    df["mass"] = df["mass"].astype(float)
    df["pseudo_mass"] = bookings["pseudo_mH"]
    df["pseudo_mass"] = df["pseudo_mass"].astype(float)
    df["weights"] = weights
    df["targets"] = labels
    df["scores"] = outputs.flatten()
    return df


# Now plotting functions
def histErrors(x, n_bins, w):
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n, bins_ = np.histogram(x, bins=bins, weights=w)
    mid = 0.5 * (bins_[1:] + bins_[:-1])
    n_err = np.sqrt(np.histogram(x, bins=bins, weights=w**2)[0])
    return n_err, n, mid


def plotScores(df_test, df_train, test_fraction, outfile):
    fig = plt.figure(figsize=(12, 8))

    x = [
        df_test.query("mass==0")["scores"],
        df_test.query("mass==400")["scores"],
        df_test.query("mass==500")["scores"],
        df_test.query("mass==600")["scores"],
        df_test.query("mass==700")["scores"],
        df_test.query("mass==800")["scores"],
        df_test.query("mass==900")["scores"],
        df_test.query("mass==1000")["scores"],
    ]

    w = [
        df_test.query("mass==0")["weights"] * (1 - test_fraction) / test_fraction,
        df_test.query("mass==400")["weights"] * (1 - test_fraction) / test_fraction * 7,
        df_test.query("mass==500")["weights"] * (1 - test_fraction) / test_fraction * 7,
        df_test.query("mass==600")["weights"] * (1 - test_fraction) / test_fraction * 7,
        df_test.query("mass==700")["weights"] * (1 - test_fraction) / test_fraction * 7,
        df_test.query("mass==800")["weights"] * (1 - test_fraction) / test_fraction * 7,
        df_test.query("mass==900")["weights"] * (1 - test_fraction) / test_fraction * 7,
        df_test.query("mass==1000")["weights"] * (1 - test_fraction) / test_fraction * 7,
    ]

    c = [
        "k",
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:cyan",
        "tab:brown",
    ]

    for i in range(0, 8):
        errs, n, mids = histErrors(x[i], 10, w[i])
        plt.errorbar(mids, n, errs, fmt="o", markersize=2, capsize=3, color=c[i])

    x = [
        df_train.query("mass==0")["scores"],
        df_train.query("mass==400")["scores"],
        df_train.query("mass==500")["scores"],
        df_train.query("mass==600")["scores"],
        df_train.query("mass==700")["scores"],
        df_train.query("mass==800")["scores"],
        df_train.query("mass==900")["scores"],
        df_train.query("mass==1000")["scores"],
    ]

    w = [
        df_train.query("mass==0")["weights"],
        df_train.query("mass==400")["weights"] * 7,
        df_train.query("mass==500")["weights"] * 7,
        df_train.query("mass==600")["weights"] * 7,
        df_train.query("mass==700")["weights"] * 7,
        df_train.query("mass==800")["weights"] * 7,
        df_train.query("mass==900")["weights"] * 7,
        df_train.query("mass==1000")["weights"] * 7,
    ]

    for i in range(0, 8):
        errs, n, mids = histErrors(x[i], 10, w[i])
        errs = np.concatenate([np.asarray([0]), errs, np.asarray([0])])
        n = np.concatenate([np.asarray([0]), n, np.asarray([0])])
        mids = np.concatenate([[-0.05], mids, [1.05]])
        # plt.fill_between(mids, n-errs, n+errs, step='mid', color=c[i], alpha=0.05) # issue with alignment
    plt.hist(
        x, weights=w, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], histtype="step", color=c, alpha=0.6
    )

    labels = [
        "Training:\n",
        r"Background",
        r"Signal $m_H=400$ GeV",
        r"Signal $m_H=500$ GeV",
        r"Signal $m_H=600$ GeV",
        r"Signal $m_H=700$ GeV",
        r"Signal $m_H=800$ GeV",
        r"Signal $m_H=900$ GeV",
        r"Signal $m_H=1000$ GeV",
        "Validation:\n",
        r"Background",
        r"Signal $m_H=400$ GeV",
        r"Signal $m_H=500$ GeV",
        r"Signal $m_H=600$ GeV",
        r"Signal $m_H=700$ GeV",
        r"Signal $m_H=800$ GeV",
        r"Signal $m_H=900$ GeV",
        r"Signal $m_H=1000$ GeV",
        r"Background",
    ]

    plt.xlabel("Classifier Score")
    plt.ylabel("SumWeights")
    plt.xlim([0, 1])

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    lines = (
        [Line2D([0], [0], color="k", linewidth=2, linestyle="None")]
        + [Line2D([0], [0], color=c_i, linewidth=2, linestyle="-") for c_i in c]
        + [Line2D([0], [0], color="k", linewidth=2, linestyle="None")]
        + [Line2D([0], [0], color=c_i, linewidth=2, marker="o", linestyle="None") for c_i in c]
    )

    plt.legend(lines, labels, ncol=2, loc="upper center")
    plt.title("Signal-Background Discrimination ge9jge3b")
    plt.savefig(outfile)


# Balance weighted data between IsSig==0/1
def BalanceSumWeights(df, weights):
    sumW_sig = sum(df.query("targets==1")[weights].tolist())
    sumW_bkg = sum(df.query("targets==0")[weights].tolist())
    w_s = 1
    w_b = float(sumW_sig) / float(sumW_bkg)
    weights = []
    for row in df["targets"].tolist():
        if row == 1:
            weights.append(w_s)
        if row == 0:
            weights.append(w_b)
    return weights


# regions 9j3b 9jge4b ge10j3b ge10jge4b


def roc_calc(df_sub, mass, split):
    if split:
        query = "pseudo_mass=={}".format(mass)
    else:
        query = "mass==0 or mass=={}".format(mass)
    weights = np.asarray(BalanceSumWeights(df_sub.query(query), "weights")) * np.asarray(
        df_sub.query(query)["weights"].tolist()
    )
    scores = df_sub.query(query)["scores"].tolist()
    targets = df_sub.query(query)["targets"].tolist()
    fpr, tpr, threshold = metrics.roc_curve(targets, scores, sample_weight=weights)

    # sample weights bug AUC calculation
    # some fpr, tpr values must be removed to ensure both curve is monotonic

    # better way to do this but it works remove entries that are not monotonic in fpr
    df_tmp = pd.DataFrame()
    df_tmp["fpr"] = fpr
    df_tmp["tpr"] = tpr
    df_tmp["fpr_sorted"] = sorted(fpr)
    df_tmp["diff"] = df_tmp["fpr"] - df_tmp["fpr_sorted"]
    df_tmp = df_tmp.drop(df_tmp.loc[df_tmp["diff"] != 0].index)
    fpr = df_tmp["fpr"].tolist()
    tpr = df_tmp["tpr"].tolist()

    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="mH={} GeV AUC = {:.3f}".format(mass, roc_auc))
    return roc_auc


def rocRegions(df, outfile, split):

    fig = plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("9j3b Region")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    aucs_9j3b = []

    df_sub = df.query("nJets==9 and nBTags==3")
    for mass in [400, 500, 600, 700, 800, 900, 1000]:
        roc_auc = roc_calc(df_sub, mass, split)
        aucs_9j3b.append(roc_auc)

    plt.legend(loc="lower right")

    plt.subplot(2, 2, 2)
    plt.title("ge10j3b Region")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    aucs_ge10j3b = []

    df_sub = df.query("nJets>=10 and nBTags==3")
    for mass in [400, 500, 600, 700, 800, 900, 1000]:
        roc_auc = roc_calc(df_sub, mass, split)
        aucs_ge10j3b.append(roc_auc)

    plt.legend(loc="lower right")
    plt.subplot(2, 2, 3)
    plt.title("9jge4b Region")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    aucs_9jge4b = []

    df_sub = df.query("nJets==9 and nBTags>=3")
    for mass in [400, 500, 600, 700, 800, 900, 1000]:
        roc_auc = roc_calc(df_sub, mass, split)
        aucs_9jge4b.append(roc_auc)

    plt.legend(loc="lower right")

    plt.subplot(2, 2, 4)
    plt.title("ge10jge4b Region")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    aucs_ge10jge4b = []

    df_sub = df.query("nJets>=10 and nBTags>=4")
    for mass in [400, 500, 600, 700, 800, 900, 1000]:
        roc_auc = roc_calc(df_sub, mass, split)
        aucs_ge10jge4b.append(roc_auc)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile)

    # Inclusive ge9jge3b region
    fig = plt.figure(figsize=(8, 6))
    plt.title("ge9jge3b Region")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    aucs_ge9jge3b = []

    df_sub = df.query("nJets>=9 and nBTags>=3")
    for mass in [400, 500, 600, 700, 800, 900, 1000]:
        roc_auc = roc_calc(df_sub, mass, split)
        aucs_ge9jge3b.append(roc_auc)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile.replace(".png", "_inclusive.png"))

    dict_aucs = {
        "ge9jge3b": aucs_ge9jge3b,
        "9j3b": aucs_9j3b,
        "ge10j3b": aucs_ge10j3b,
        "9jge4b": aucs_9jge4b,
        "ge10jge4b": aucs_ge10jge4b,
    }
    return dict_aucs


# Plot auc output of previous function for each region
def aucRegions(aucs_train, aucs_test, outfile):
    masses = [400, 500, 600, 700, 800, 900, 1000]

    fig = plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("9j3b Region")
    plt.ylabel("AUC")
    plt.xlabel("Mass")

    plt.plot(
        masses,
        aucs_test["9j3b"],
        label="Validation",
        marker="o",
        linestyle="dashed",
    )
    plt.plot(
        masses,
        aucs_train["9j3b"],
        label="Training",
        marker="o",
        linestyle="dashed",
    )

    plt.legend(loc="lower right")

    plt.subplot(2, 2, 2)
    plt.title("ge10j3b Region")
    plt.ylabel("AUC")
    plt.xlabel("Mass")

    plt.plot(
        masses,
        aucs_test["ge10j3b"],
        label="Validation",
        marker="o",
        linestyle="dashed",
    )
    plt.plot(
        masses,
        aucs_train["ge10j3b"],
        label="Training",
        marker="o",
        linestyle="dashed",
    )

    plt.legend(loc="lower right")

    plt.subplot(2, 2, 3)
    plt.title("9jge4b Region")
    plt.ylabel("AUC")
    plt.xlabel("Mass")

    plt.plot(
        masses,
        aucs_test["9jge4b"],
        label="Validation",
        marker="o",
        linestyle="dashed",
    )
    plt.plot(
        masses,
        aucs_train["9jge4b"],
        label="Training",
        marker="o",
        linestyle="dashed",
    )

    plt.legend(loc="lower right")

    plt.subplot(2, 2, 4)
    plt.title("ge10jge4b Region")

    plt.ylabel("AUC")
    plt.xlabel("Mass")

    plt.plot(
        masses,
        aucs_test["ge10jge4b"],
        label="Validation",
        marker="o",
        linestyle="dashed",
    )
    plt.plot(
        masses,
        aucs_train["ge10jge4b"],
        label="Training",
        marker="o",
        linestyle="dashed",
    )

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()

    # plot inclusive
    fig = plt.figure(figsize=(8, 6))
    plt.title("ge9jge3b Region")

    plt.ylabel("AUC")
    plt.xlabel("Mass")

    plt.plot(
        masses,
        aucs_test["ge9jge3b"],
        label="Validation",
        marker="o",
        linestyle="dashed",
    )
    plt.plot(
        masses,
        aucs_train["ge9jge3b"],
        label="Training",
        marker="o",
        linestyle="dashed",
    )

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile.replace(".png", "_inclusive.png"))
    plt.show()


def pseudoMassDistribution(df_scaled, outfile):

    pseudo_frac_bkg = [
        sum(df_scaled.query("mH_label==0 and pseudo_mH_unscaled==400")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==0 and pseudo_mH_unscaled==500")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==0 and pseudo_mH_unscaled==600")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==0 and pseudo_mH_unscaled==700")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==0 and pseudo_mH_unscaled==800")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==0 and pseudo_mH_unscaled==900")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==0 and pseudo_mH_unscaled==1000")["sample_weight"].tolist()),
    ]

    pseudo_frac_sig = [
        sum(df_scaled.query("mH_label==400")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==500")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==600")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==700")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==800")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==900")["sample_weight"].tolist()),
        sum(df_scaled.query("mH_label==1000")["sample_weight"].tolist()),
    ]

    masses = [400, 500, 600, 700, 800, 900, 1000]

    plt.bar(masses, pseudo_frac_sig, label="Signal", width=50, alpha=0.8, align="edge")
    plt.bar(masses, pseudo_frac_bkg, label="Background", width=50, alpha=0.8)
    plt.xlabel("Pseudo-mass")
    plt.ylabel("SumWeights")
    plt.title("Weighted pseudo-mass distribution")
    plt.legend(loc="lower right")
    plt.savefig(outfile)


def plotAll(df_train, df_test, test_fraction, outfile_prefix):
    os.system("mkdir -p {}".format(outfile_prefix))
    # Plot scores
    plotScores(df_test, df_train, test_fraction, "{}/scoreHistogram.png".format(outfile_prefix))
    # Plot roc curves split by pseudo_mass
    print("Splitting by background by pseudo_mass")
    split = True
    aucs_test_split = rocRegions(df_test, "{}/ROC_Test_split.png".format(outfile_prefix), split)
    aucs_train_split = rocRegions(df_train, "{}/ROC_Train_split.png".format(outfile_prefix), split)
    # Plot AUCs for each region
    aucRegions(aucs_train_split, aucs_test_split, "{}/AUC_regions_split.png".format(outfile_prefix))

    return aucs_train_split, aucs_test_split


def plotGraph(G, node_scale, global_features):
    color_map = []
    pos = {}
    pt_max = 0
    # sum_pt=np.asarray([0,0])
    for node in G:
        pt = G.nodes[node]["features"][0]
        btag = G.nodes[node]["features"][4]
        if node == 0:
            color_map.append(cm.bone(0))
        # else:
        # sum_pt=sum_pt+np.asarray([pt*np.cos(G.nodes[node]['pos'][1]), pt*np.sin(G.nodes[node]['pos'][1])])
        if node == 1:
            color_map.append(cm.bwr(0))
        if node > 1:
            color_map.append(cm.summer(btag))

        pt_max = max(pt, pt_max)
        pos[node] = (
            pt * np.cos(node_scale[1] * G.nodes[node]["features"][1]),
            pt * np.sin(node_scale[1] * G.nodes[node]["features"][1]),
        )

    ax = plt.figure(figsize=(10, 10)).gca()

    t = 2 * np.pi * np.linspace(0, 1, 1000)

    plt.plot(pt_max * np.cos(t), pt_max * np.sin(t))
    plt.plot(0.01 * pt_max * np.cos(t), 0.01 * pt_max * np.sin(t), "red")

    nx.draw(G, ax=ax, pos=pos, node_color=color_map, alpha=0.7)

    print("Book-keeping variables:\n")
    for key in G.graph["book_keeping"].keys():
        print("{}: ".format(key), G.graph["book_keeping"][key])
    print("\n")

    global_dict = dict(zip(global_features, G.graph["features"]))

    print("Global variables:\n")
    for key in global_dict.keys():
        print("{}: ".format(key), global_dict[key])
    print("\n")

    print("Sample weight: ", G.graph["sample_weight"])
    print("Pseudo-mass:", G.graph["pseudo_mH"])
    print("Target: ", G.graph["IsSig"])
    print("\n")
    for node in G:
        print(
            "Node: {}".format(node),
            "\tpT: {:.4f}".format(G.nodes[node]["features"][0]),
            "\tphi: {:.4f}".format(G.nodes[node]["features"][1]),
            "\teta: {:.4f}".format(G.nodes[node]["features"][2]),
            "\tE: {:.4f}".format(G.nodes[node]["features"][3]),
            "\tbtag: {:.1f}".format(G.nodes[node]["features"][4]),
            "\tQ/G-BDT: {:.4f}".format(G.nodes[node]["features"][5]),
            "\ttype: {}".format(G.nodes[node]["features"][6]),
        )
