import ROOT
import numpy as np
import time
from itertools import combinations, permutations
import torch

print("pyTorch Version: ", torch.__version__)
from torch_geometric.data import Dataset, Data, DataLoader
import networkx as nx

# Imports for GNN model, may need to add more depending on model object class
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Dropout, Tanh
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn import MetaLayer, global_mean_pool, global_max_pool, global_add_pool


# from pymva.extra import CloneFile
# todo integrate with tmva-kit, copy-pasted function for now
from ROOT import TFile, TTree, vector, TCut, std
from array import array


def CloneFile(path, fil, tree_names, y_pred, var_name="score", ntup_opt="recreate", same_path=False):
    print(("FileName to be read: %s") % fil)
    tfile = TFile(fil)
    trees = []

    if len(tree_names) != len(y_pred):
        print("Number of trees and number of prediction must be equal")
        exit()

    for t in tree_names:
        print("Tree will be cloned: %s" % t)
        trees.append(tfile.Get(t))

    score = vector("float")()
    print("\nUpdating File --------------------------")
    fil_new = fil.replace(".root", "_clone.root")
    if not same_path:
        fil_new = path + fil[fil.rfind("/") + 1 :].replace(".root", "_clone.root")
    print(("FileName to be recorded: %s") % fil_new)
    trees_new = []
    tfile_new = TFile(fil_new, ntup_opt)
    for t in trees:
        trees_new.append(t.CloneTree())
        trees_new[-1].Branch(var_name, score)

    for i in range(len(trees_new)):
        for x in y_pred[i]:
            score.clear()
            if not np.isscalar(x):
                for e in x:
                    score.push_back(e)
            else:
                score.push_back(x)
            trees_new[i].GetBranch(var_name).Fill()
        trees_new[i].Write()
    tfile_new.Close()
    print("Closing File --------------------------\n")
    return fil_new


def root2pandas(file_i, tree_name, **kwargs):
    from root_numpy import root2array

    df = pd.DataFrame(root2array(file_i, tree_name, **kwargs).view(np.recarray))
    return df


# add extra vars to dataframe


def add_HT_ratio(df):
    ht_ratio = []
    for index, event in df.iterrows():
        ht_ratio.append(sum(np.asarray(event["jet_pt"])[0:3]) / sum(np.asarray(event["jet_pt"])[4:-1]))
    df["HT_ratio"] = ht_ratio
    return df


def add_sum_rcjets_d12_d23(df):
    sum_d12 = []
    sum_d23 = []
    for index, event in df.iterrows():
        sum_d12.append(sum(event["rcjet_d12"]))
        sum_d23.append(sum(event["rcjet_d23"]))
    df["Sum_rcjet_d12"] = sum_d12
    df["Sum_rcjet_d23"] = sum_d23
    return df


def calibrate_qg_tagging(df):
    # construct qg_bdt calibration cut on bjets, this is time consuming
    jet_qg_BDT_calibrated_list = []
    jet_qg_ngjets_list = []
    for index, row in df[["jet_tagWeightBin_DL1r_Continuous", "jet_qg_BDT", "jet_eta"]].copy().iterrows():
        if index % 1000 == 0 or index == len(df) - 1:
            print("\r{:.1f} %".format(index / len(df) * 100), end="")

        row["jet_tagWeightBin_DL1r_Continuous_copy"] = row["jet_tagWeightBin_DL1r_Continuous"].copy()
        conditional_bjets = np.asarray(row.copy()["jet_tagWeightBin_DL1r_Continuous_copy"])
        conditional_bjets[conditional_bjets < 2] = 1
        conditional_bjets[conditional_bjets >= 2] = 0

        row["jet_qg_BDT_copy"] = row["jet_qg_BDT"].copy()
        conditional_gjets = np.asarray(row["jet_qg_BDT_copy"])
        conditional_gjets[conditional_gjets > -0.02] = 1
        conditional_gjets[conditional_gjets <= -0.02] = 0

        row["jet_eta_copy"] = row["jet_eta"].copy()
        conditional_eta = np.asarray(row["jet_eta_copy"])
        conditional_eta[abs(conditional_eta) < 2.1] = 1
        conditional_eta[abs(conditional_eta) >= 2.1] = 0

        conditional = conditional_bjets * conditional_gjets * conditional_eta

        jet_qg_BDT_calibrated_list.append(np.asarray(row["jet_qg_BDT"]) * conditional)
        jet_qg_ngjets_list.append(sum(conditional))

    df["jet_qg_BDT_calibrated"] = pd.Series(jet_qg_BDT_calibrated_list)
    df["jet_qg_BDT_ngTags"] = pd.Series(jet_qg_ngjets_list)
    return df


# convert data frame element to networkx graph
def event2networkx(event, global_features, global_scale, node_scale, include_node_qg_bdt=True):

    # create empty graph
    G = nx.DiGraph()
    # event graph is made up of global variables and objects
    # objects are jets, electrons, muons etc they have attributes such as pT, phi, eta
    # objects are assigned to nodes, edges are generated from differences between nodes in phi, eta space

    bookkeeping_features = ["eventNumber", "runNumber", "mcChannelNumber", "nBTags_DL1r_70", "nJets"]

    # global variables are assigned to graph features
    G.graph["features"] = np.asarray(event[global_features].tolist()) / global_scale
    G.graph["book_keeping"] = dict(zip(bookkeeping_features, event[bookkeeping_features].tolist()))

    index = 0

    # add met as an object node with eta=0
    met_encoding = [0]
    met_features = np.asarray(event[["met_met", "met_phi"]].tolist() + [0] + [event["met_met"]] + [0, 0] + met_encoding)
    G.add_node(index, features=met_features / node_scale)

    # loop through jets and leptons to assign
    index = index + 1

    electron_features = ["el_pt", "el_phi", "el_eta", "el_e"]
    nElectrons = len(event["el_pt"])
    nfeatures = len(electron_features)
    electrons = np.vstack(event[electron_features])
    electron_encoding = [-1]
    for i in range(0, nElectrons):
        electron = list((electrons[0:nfeatures, i]).flatten()) + [0, 0] + electron_encoding
        G.add_node(index, features=np.asarray(electron) / node_scale)
        index = index + 1

    muon_features = ["mu_pt", "mu_phi", "mu_eta", "mu_e"]
    nMuons = len(event["mu_pt"])
    nfeatures = len(muon_features)
    muons = np.vstack(event[muon_features])
    muon_encoding = [-2]
    for i in range(0, nMuons):
        muon = list((muons[0:nfeatures, i]).flatten()) + [0, 0] + muon_encoding
        G.add_node(index, features=np.asarray(muon) / node_scale)
        index = index + 1

    jet_features = [
        "jet_pt",
        "jet_phi",
        "jet_eta",
        "jet_e",
        "jet_tagWeightBin_DL1r_Continuous",
        "jet_qg_BDT_calibrated",
    ]
    nJets = event["nJets"]
    nfeatures = len(jet_features)
    jets = np.vstack(event[jet_features])
    jet_encoding = [1]
    for i in range(0, nJets):
        jet = list((jets[0:nfeatures, i]).flatten()) + jet_encoding
        G.add_node(index, features=np.asarray(jet) / node_scale)
        index = index + 1

    # now need to add edges
    # get pairs of objects and calculate difference in phi an eta
    objects = list(G.nodes)
    pairs = permutations(objects, 2)
    # loop through pairs and calculate delta_phi delta_eta
    for pair in pairs:
        # take difference between phi and eta of two nodes, could add DeltaR
        delta_phi = (
            np.arctan2(
                np.sin(node_scale[1] * (G.nodes[pair[0]]["features"][1] - G.nodes[pair[1]]["features"][1])),
                np.cos(node_scale[1] * (G.nodes[pair[0]]["features"][1] - G.nodes[pair[1]]["features"][1])),
            )
            / node_scale[1]
        )  # this is computationally expensive but give correct angle and sign
        delta_eta = G.nodes[pair[0]]["features"][2] - G.nodes[pair[1]]["features"][2]
        delta_R = np.sqrt((node_scale[1] * delta_phi) ** 2 + (node_scale[2] * delta_eta) ** 2) / (
            node_scale[1] + node_scale[2]
        )
        G.add_edge(pair[0], pair[1], features=np.asarray([delta_phi, delta_eta, delta_R]))

    return G


def getNodeFeatures(G):
    x = np.zeros(shape=(G.number_of_nodes(), len(G.nodes[0]["features"])))
    for node in G.nodes:
        x[int(node)] = np.asarray(G.nodes[node]["features"])
    return torch.from_numpy(x)


def getEdgeFeatures(G):
    e = np.zeros(shape=(G.number_of_edges(), len(G.edges[0, 1]["features"])))
    i = 0
    for edge in G.edges:
        e[i] = np.asarray(G.edges[edge]["features"])
        i = i + 1
    return torch.from_numpy(e)


def getEdgeList(G):
    return torch.from_numpy(np.asarray(G.edges())).t().contiguous().long()


# function to create pytorch-geometric data object
# takes pandas dataframe as input
def CreateTorchGraphs(data, global_features, global_scale, node_scale, remove_phi=True, include_qg_bdt=True):
    print("Creating graph data...")
    graphs = []
    booking = []
    i = 0
    start = time.time()
    for index, event in data.reset_index(drop=True).iterrows():
        i += 1
        if i % 100 == 0 or i == len(data):
            elapsed = time.time() - start
            graphs_per_second = i / elapsed
            graphs_remaining = len(data) - i
            seconds_remaining = graphs_remaining / graphs_per_second
            print(
                "\r{}/{} complete. Time elapsed: {:.1f}s,   Estimated time remaining: {:.1f}s".format(
                    i, len(data), elapsed, seconds_remaining
                ),
                end="",
            )
        G_nx = event2networkx(event, global_features, global_scale, node_scale, include_qg_bdt)
        x = getNodeFeatures(G_nx)
        edge_attr = getEdgeFeatures(G_nx)
        if remove_phi:
            if include_qg_bdt:
                node_keep_indices = [0, 2, 3, 4, 5, 6]
            else:
                node_keep_indices = [0, 2, 3, 4, 5]
            x = x[:, node_keep_indices]

        edge_index = getEdgeList(G_nx)
        G_geo = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=torch.from_numpy(np.asarray(G_nx.graph["features"])).float().view(-1, len(G_nx.graph["features"])),
            pseudo_mH=torch.tensor(G_nx.graph["pseudo_mH"]),
        )
        graphs.append(G_geo)
        booking.append(G_nx.graph["book_keeping"])
    print("\nDone")
    return graphs, pd.DataFrame(booking).reset_index(drop=True)


# Overload torch.geometric Dataset class to include our own graphs
class customDataset(Dataset):
    def __init__(self, graphs=[]):
        super(customDataset, self).__init__()
        self.graphs = graphs
        self.n = 10

    def save_to(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        remainder_n = len(self.graphs) % self.n
        len_int = int(len(self.graphs) - remainder_n)
        for i in range(self.n):
            if i == self.n - 1:
                end = len(graphs)
            else:
                end = int((i * len_int + len_int) / self.n)
            start = int(i * len_int / self.n)
            with open("{}/graphs_{}.pkl".format(path, i), "wb") as f:
                pickle.dump(self.graphs[start:end], f)

    def download_from(self, path):
        self.booking = pd.DataFrame()
        self.graphs = []
        for i in range(self.n):
            print("Downloading file {}/{}...".format(i + 1, self.n))
            with open("{}/graphs_{}.pkl".format(path, i), "rb") as f:
                self.graphs = self.graphs + pickle.load(f)
        print("Done")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# Define function to return gnn scores for a given model, data and pseudo-mass input.
def get_gnn_scores(model, loader, pseudo_mass):
    model.eval()
    outputs = []
    for data in loader:
        tensor = torch.zeros(1)
        pseudo_mass_input = tensor.new_full(data.pseudo_mass.shape, pseudo_mass)  # Define pseudo-mass as input
        output = model(data.x, data.edge_index, data.edge_attr, data.u, pseudo_mass_input, data.batch).detach()
        outputs.append(output)
    return np.concatenate(outputs, axis=0).flatten()


# Define function to loop over pseudo-mass values and get all scores.  Note: important that DataLoader provided has shuffle=False
def get_all_pseudomass_scores(model, loader):
    masses = [400, 500, 600, 700, 800, 900, 1000]  # GeV
    outputs = np.empty(shape=(len(masses), len(loader.dataset)), dtype=float)
    outputs.fill(np.nan)
    i = 0
    for mass in masses:
        print("Evaluating GNN scores for pseudo-mass input: {} GeV".format(mass))
        output = get_gnn_scores(model, loader, mass / 1000)  # Pseudo-mass weighting is hardcoded
        outputs[i] = output
        i += 1
    return np.transpose(outputs)


# Define function to get ntuple, create graphs, evaluate gnn scores and append to ntuple
def eval_append_ntuple(ntuple, tree_name, outpath, model, global_features, global_scale, node_scale):
    print("Reading from file: ", ntuple)
    variables_to_read = [
        "jet_pt",
        "jet_phi",
        "jet_eta",
        "jet_e",
        "jet_tagWeightBin_DL1r_Continuous",
        "jet_qg_BDT",
        "mu_pt",
        "mu_phi",
        "mu_eta",
        "mu_e",
        "el_pt",
        "el_phi",
        "el_eta",
        "el_e",
        "nJets",
        "met_met",
        "met_phi",
        "eventNumber",
        "runNumber",
        "mcChannelNumber",
        "nBTags_DL1r_70",
    ] + global_features
    df = root2pandas(ntuple, tree_name, branches=variables_to_read)
    # need to add extra vars
    df = add_HT_ratio(df)
    df = add_sum_rcjets_d12_d23(df)
    df = calibrate_qg_tagging(df)
    graphs, booking = CreateTorchGraphs(
        data, global_features, global_scale, node_scale, remove_phi=True, include_qg_bdt=True
    )
    dataset = customDataset(graphs)  # Store graphs in custom pyTorch dataset
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)  # Create DataLoader to handle batches
    print("Evaluating GNN classification...")
    outputs = get_all_pseudomass_scores(model, loader)
    print("Done!")
    print("Cloning ntuple with GNN scores appended...")
    new_file = CloneFile(outpath + "/", ntuple, [tree_name], [outputs], "GNN_Score", "recreate", same_path=False)


import pickle

# Define function to load model from object class.pkl and saved weights.pt
def load_model(model_class, model_weights):
    print("Loading pyTorch model. Model architecture: {}, Model weights: {}".format(model_class, model_weights))
    with open(model_class, "rb") as pic:
        model = pickle.load(pic)  # Load object class from pickle
    model = torch.load(model_weights, map_location=torch.device("cpu"))  # Load weights
    return model


# Parse args
import argparse

parser = argparse.ArgumentParser(
    description="Evaluate ntuples with GNN model and append classification scores to cloned ntuples."
)
parser.add_argument("--ntuples", nargs="+", help="Ntuples to process.", action="store", required=True)
parser.add_argument("--config", help="Path to config file.", action="store", required=True)
parser.add_argument("--outdir", help="Path to output directory.", action="store", required=True)
parser.add_argument("--tree-name", help="Tree name to evaluate.", action="store", required=True)
parser.add_argument("--model", help="Path to model architecture instance.")
parser.add_argument("--weights", help="Path to model weights.")
args = parser.parse_args()

# Do everything
def cloneNtuples(args):
    config = args.config.replace(".py", "")
    from config import global_features, global_scale, node_scale  # load in feature names and scaling from config
    from config import GeneralMPGNN, EdgeModel, NodeModel, GlobalModel  # import class definitions from config
    from config import include_qg_bdt, remove_phi  # import options for building graphs

    print("Using global features: ", global_features)
    print("Using global feature scale: ", global_scale)
    print("Using node feature scale: ", node_scale)
    model = load_model(args.model, args.weights)  # Load model
    # Loop over ntuples
    for ntuple in args.ntuples:
        # Create graphs, evaluate GNN scores and clone ntuple w/ scores
        eval_append_ntuple(ntuple, args.tree_name, args.outdir, model, global_features, global_scale, node_scale)
    print("Done!")


# Define function to convert event (from Ttree for loop) to graph
def event2ptgeo(event, global_vars, global_scale, node_scale, label, pseudo_mass):
    # Global variable
    globals_ = torch.tensor([getattr(event, global_var) for global_var in global_vars]) / global_scale
    # Node features are pt, phi, eta, e, btag, object type encoding
    # Node feature names are hardcoded for now, configurable in future
    # Encoding can be continuous or one-hot, configurable in future but hard-coded as continuous for now
    nodes = []

    # MET
    # Note for MET node: eta=0, btag=0, pt=e
    met_met = getattr(event, "met_met")
    met_phi = getattr(event, "met_phi")
    met_encoding = [0]
    met_node = [met_met, met_phi, 0, met_met, 0] + met_encoding
    nodes.append(list(np.asarray(met_node) / node_scale))

    # Leptons
    # Note for lepton nodes btag=0
    el_pt = list(getattr(event, "el_pt"))
    el_phi = list(getattr(event, "el_phi"))
    el_eta = list(getattr(event, "el_eta"))
    el_e = list(getattr(event, "el_e"))
    el_encoding = [-1]
    nElectrons = len(el_phi)
    if nElectrons != 0:
        for i in range(nElectrons):
            el_node = [el_pt[i], el_phi[i], el_eta[i], el_e[i], 0] + el_encoding
            nodes.append(list(np.asarray(el_node) / node_scale))

    mu_pt = list(getattr(event, "mu_pt"))
    mu_phi = list(getattr(event, "mu_phi"))
    mu_eta = list(getattr(event, "mu_eta"))
    mu_e = list(getattr(event, "mu_e"))
    mu_encoding = [-2]
    nMuons = len(mu_phi)
    if nMuons != 0:
        for i in range(nMuons):
            mu_node = [mu_pt[i], mu_phi[i], mu_eta[i], mu_e[i], 0] + mu_encoding
            nodes.append(list(np.asarray(mu_node) / node_scale))
    # Jets
    jets_pt = list(getattr(event, "jet_pt"))
    jets_phi = list(getattr(event, "jet_phi"))
    jets_eta = list(getattr(event, "jet_eta"))
    jets_e = list(getattr(event, "jet_e"))
    jets_btag = list(getattr(event, "jet_tagWeightBin_MV2c10_Continuous"))
    jets_encoding = [1]
    nJets = len(jets_btag)
    for i in range(nJets):
        jet_node = [jets_pt[i], jets_phi[i], jets_eta[i], jets_e[i], jets_btag[i]] + jets_encoding
        nodes.append(list(np.asarray(jet_node) / node_scale))

    # Convert list of lists containing node features to tensor
    x = torch.tensor(nodes)
    n_nodes = len(nodes)

    # Get edge pairs as tuple list
    edge_coordinate_tuples = list(
        permutations(range(n_nodes), 2)
    )  # Use permutations for directed graph, combinations for undirected
    # Convert to COO tensor format, note edge index must be type long
    edge_index = torch.from_numpy(np.array(edge_coordinate_tuples)).t().contiguous().long()

    # Fill edges with attributes
    edge_attr = []
    n_edges = edge_index.shape[1]
    # Create edges
    for i in range(n_edges):
        row, col = edge_index[0, i], edge_index[1, i]
        delta_phi = (
            np.arctan2(np.sin(node_scale[1] * (x[row][1] - x[col][1])), np.cos(node_scale[1] * (x[row][1] - x[col][1])))
            / node_scale[1]
        )
        delta_eta = x[row][2] - x[col][2]
        edge_attr.append([delta_phi, delta_eta])
    edge_attr = torch.tensor(edge_attr)

    # Assign info to graph object
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=globals_.view(-1, len(globals_)),
        y=torch.tensor(label).float(),
        pseudo_mass=torch.tensor(pseudo_mass),
    )
    return graph


# Define function to convert tree to list of graphs
def tree2graphs_eval(tree, global_vars, global_scale, node_scale):
    graphs = []
    n_entries = tree.GetEntries()
    i = 0
    start = time.time()
    # Loop through events in tree, convert to graph, append graph to list
    for event in tree:
        i += 1
        if i % 100 == 0 or i == n_entries:
            elapsed = time.time() - start
            graphs_per_second = i / elapsed
            graphs_remaining = n_entries - i
            seconds_remaining = graphs_remaining / graphs_per_second
            print(
                "\rCreating graphs. {}/{} complete. Time elapsed: {:.5}s,   Estimated time remaining: {:.5}s".format(
                    i, n_entries, elapsed, seconds_remaining
                ),
                end="",
            )
        graph = event2ptgeo(
            event, global_vars, global_scale, node_scale, -999, -999
        )  # Will define pseudo-mass during evaluation later, label doesn't matter
        graphs.append(graph)
    print("\nDone!")
    return graphs


# from pymva.extra import CloneFile
# todo integrate with tmva-kit, copy-pasted function for now
from ROOT import TFile, TTree, vector, TCut, std
from array import array


def CloneFile(path, fil, tree_names, y_pred, var_name="score", ntup_opt="recreate", same_path=False):
    print(("FileName to be read: %s") % fil)
    tfile = TFile(fil)
    trees = []

    if len(tree_names) != len(y_pred):
        print("Number of trees and number of prediction must be equal")
        exit()

    for t in tree_names:
        print("Tree will be cloned: %s" % t)
        trees.append(tfile.Get(t))

    score = vector("float")()
    print("\nUpdating File --------------------------")
    fil_new = fil.replace(".root", "_clone.root")
    if not same_path:
        fil_new = path + fil[fil.rfind("/") + 1 :].replace(".root", "_clone.root")
    print(("FileName to be recorded: %s") % fil_new)
    trees_new = []
    tfile_new = TFile(fil_new, ntup_opt)
    for t in trees:
        trees_new.append(t.CloneTree())
        trees_new[-1].Branch(var_name, score)

    for i in range(len(trees_new)):
        for x in y_pred[i]:
            score.clear()
            if not np.isscalar(x):
                for e in x:
                    score.push_back(e)
            else:
                score.push_back(x)
            trees_new[i].GetBranch(var_name).Fill()
        trees_new[i].Write()
    tfile_new.Close()
    print("Closing File --------------------------\n")
    return fil_new
