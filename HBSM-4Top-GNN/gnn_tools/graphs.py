from torch_geometric.data import Data
import time
import networkx as nx
import numpy as np
import pandas as pd
from itertools import combinations, permutations
import torch
from torch_geometric.data import Dataset
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps as cm


def event2networkx(event, global_features, global_scale, node_scale, edge_scale):
    """
    This function takes an event (as DataFrame row) and generates a networkx graph. Global features, book keeping variables, sample weight, IsSig, pseudo_mH are added as global graph attributes. Function iterates through individual objects (MET, muons, electrons, jets) and extracts node features [pt, phi, eta (0 for MET), E, pcb_score (0 for non-jets), qg_bdt_score (0 for non-jets)] an object type encoding is assigned for each object (MET:0, electron:-1, muon:-2, jet:1). Objects features are added to nodes. Node and global features are scaled by arrays of scaling coefficients. For each pair of nodes in the graph an edge is constructed and edge features (delta_phi, delta_eta and delta_R) are assigned. Currently node features are hard-coded.

    Inputs:
      event (Pandas DataFrame row): Row should contain columns such as jet_pt, jet_phi etc. For objects with multiplicity, row entries should be lists containing an entry for each object of that type ordered by pt.
      global_features (list of strings): Global features to assign to graph.
      global_scale (numpy array): Scaling coefficients to scale global features by.
      node_scale (numpy array): Scaling coefficients to scale node features by.

    Returns:
      G (networkx graph object): A graph containing global, node and edge features which represents an event.
    """
    # create empty graph
    G = nx.DiGraph()
    # event graph is made up of global variables and objects
    # objects are jets, electrons, muons etc they have attributes such as pT, phi, eta
    # objects are assigned to nodes, edges are generated from differences between nodes in phi, eta space

    bookkeeping_features = [
        "eventNumber",
        "runNumber",
        "mcChannelNumber",
        "mH_label",
        "pseudo_mH",
        "nBTags_DL1r_70",
        "nJets",
    ]

    # global variables are assigned to graph features
    G.graph["features"] = np.asarray(event[global_features].tolist()) / global_scale
    G.graph["book_keeping"] = dict(zip(bookkeeping_features, event[bookkeeping_features].tolist()))
    G.graph["sample_weight"] = event["sample_weight"]
    G.graph["IsSig"] = event["IsSig"]
    G.graph["pseudo_mH"] = event["pseudo_mH"] / 1000

    index = 0

    # add met as an object node with eta=0
    met_encoding = 0
    met_features = np.asarray(event[["met_met", "met_phi"]].tolist() + [0, event["met_met"], 0, met_encoding])
    G.add_node(index, features=met_features / node_scale)

    # loop through jets and leptons to assign
    index = index + 1

    electron_features = ["el_pt", "el_phi", "el_eta", "el_e"]
    nElectrons = len(event["el_pt"])
    nfeatures = len(electron_features)
    electrons = np.vstack(event[electron_features])
    electron_encoding = -1
    for i in range(0, nElectrons):
        electron = list((electrons[0:nfeatures, i]).flatten()) + [0, +electron_encoding]
        G.add_node(index, features=np.asarray(electron) / node_scale)
        index = index + 1

    muon_features = ["mu_pt", "mu_phi", "mu_eta", "mu_e"]
    nMuons = len(event["mu_pt"])
    nfeatures = len(muon_features)
    muons = np.vstack(event[muon_features])
    muon_encoding = -2
    for i in range(0, nMuons):
        muon = list((muons[0:nfeatures, i]).flatten()) + [0, muon_encoding]
        G.add_node(index, features=np.asarray(muon) / node_scale)
        index = index + 1

    jet_features = ["jet_pt", "jet_phi", "jet_eta", "jet_e", "jet_tagWeightBin_DL1r_Continuous"]
    nJets = event["nJets"]
    nfeatures = len(jet_features)
    jets = np.vstack(event[jet_features])
    jet_encoding = 1
    for i in range(0, nJets):
        jet = list((jets[0:nfeatures, i]).flatten()) + [jet_encoding]
        G.add_node(index, features=np.asarray(jet) / node_scale)
        index = index + 1

    # now need to add edges
    # get pairs of objects and calculate difference in phi an eta
    objects = list(G.nodes)
    pairs = permutations(objects, 2)
    # loop through pairs and calculate delta_phi delta_eta
    for pair in pairs:
        # take difference between phi and eta of two nodes, could add DeltaR
        delta_phi = np.arctan2(
            np.sin(node_scale[1] * (G.nodes[pair[0]]["features"][1] - G.nodes[pair[1]]["features"][1])),
            np.cos(node_scale[1] * (G.nodes[pair[0]]["features"][1] - G.nodes[pair[1]]["features"][1])),
        )  # this is computationally expensive but give correct angle and sign
        delta_eta = node_scale[2] * (G.nodes[pair[0]]["features"][2] - G.nodes[pair[1]]["features"][2])
        delta_R = np.sqrt((node_scale[1] * delta_phi) ** 2 + (node_scale[2] * delta_eta) ** 2)
        G.add_edge(pair[0], pair[1], features=np.asarray([delta_phi, delta_eta, delta_R]) / edge_scale)

    return G


def CreateTorchGraphsNetworkx(data, global_features, global_scale, node_scale, edge_scale):
    """
    Function to loop through events and create graphs.

    Inputs:
      data (Pandas DataFrame): Row should contain an event with columns such as jet_pt, jet_phi etc. For objects with multiplicity, row entries should be lists containing an entry for each object of that type ordered by pt.
      global_features (list of strings): Global features to assign to graph.
      global_scale (numpy array): Scaling coefficients to scale global features by.
      node_scale (numpy array): Scaling coefficients to scale node features by.

    Returns:
      graphs (list): List of PyTorch Data objects containing graphs.
      df_booking (Pandas DataFrame): DataFrame of book-keeping information to identify graphs in future.
    """
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
        G_nx = event2networkx(event, global_features, global_scale, node_scale, edge_scale)
        keep_indices = [0, 2, 3, 4, 5]  # delete phi
        x = getNodeFeatures(G_nx)[:, keep_indices]
        edge_attr = getEdgeFeatures(G_nx)
        edge_index = getEdgeList(G_nx)
        G_geo = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.from_numpy(np.asarray(G_nx.graph["IsSig"])).float(),
            w=torch.from_numpy(np.asarray(G_nx.graph["sample_weight"])).float(),
            u=torch.from_numpy(np.asarray(G_nx.graph["features"])).float().view(-1, len(G_nx.graph["features"])),
            pseudo_mH=torch.tensor(G_nx.graph["pseudo_mH"]),
        )
        graphs.append(G_geo)
        booking.append(G_nx.graph["book_keeping"])
    df_booking = pd.DataFrame(booking).reset_index(drop=True)
    print("\nDone")
    return graphs, df_booking


def lorentzVector(pt, phi, eta, e):
    return np.array([e, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)])


def event2Tensor(event, nodeScale, edgeScale):
    # nodes
    # jets
    jet_features = ["jet_pt", "jet_phi", "jet_eta", "jet_e", "jet_tagWeightBin_DL1r_Continuous"]
    nJets = event["nJets"]
    jet_encoding = 1
    jets = np.vstack(event[jet_features])  # first index is object, second is feature
    jets = np.append(jets, jet_encoding * np.ones(nJets).reshape(1, nJets), axis=0)

    # muons
    muon_features = ["mu_pt", "mu_phi", "mu_eta", "mu_e"]
    nMuons = event["nMuons"]
    muon_encoding = -2
    muons = np.vstack(event[muon_features])  # first index is object, second is feature
    muons = np.append(muons, np.array([np.zeros(nMuons), muon_encoding * np.ones(nMuons)]), axis=0)

    # electrons
    electron_features = ["el_pt", "el_phi", "el_eta", "el_e"]
    nElectrons = event["nElectrons"]
    electron_encoding = -1
    electrons = np.vstack(event[electron_features])  # first index is object, second is feature
    electrons = np.append(electrons, np.array([np.zeros(nElectrons), electron_encoding * np.ones(nElectrons)]), axis=0)

    # met
    nNodeFeats = len(jet_features) + 1
    met_encoding = 0
    met = np.asarray(event[["met_met", "met_phi"]].tolist() + [0] + [event["met_met"]] + [0, met_encoding]).reshape(
        nNodeFeats, 1
    )

    nodes = np.transpose(np.hstack([met, jets, electrons, muons]))

    # edges
    objects = range(len(nodes))
    pairs = permutations(objects, 2)
    edges = []
    pairs_list = []
    for pair in pairs:
        pairs_list.append(list(pair))  # store this for making edge index
        delta_phi = np.arctan2(
            np.sin(nodes[pair[0]][1] - nodes[pair[1]][1]), np.cos(nodes[pair[0]][1] - nodes[pair[1]][1])
        )
        delta_eta = nodes[pair[0]][2] - nodes[pair[1]][2]
        delta_R = np.sqrt(delta_phi**2 + delta_eta**2)

        # v0=lorentzVector(nodes[pair[0]][0],nodes[pair[0]][1],nodes[pair[0]][2],nodes[pair[0]][3])
        # v1=lorentzVector(nodes[pair[1]][0],nodes[pair[1]][1],nodes[pair[1]][2],nodes[pair[1]][3])
        # v12=v0+v1
        # M_inv=np.sqrt(v12[0]**2-v12[1]**2-v12[2]**2-v12[3]**2)

        edges.append([delta_phi, delta_eta, delta_R])
    edge = np.asarray(edges)

    # index
    edgeIndex = torch.from_numpy(np.asarray(pairs_list)).t().contiguous().long()

    return torch.from_numpy(nodes / nodeScale), torch.from_numpy(edges / edgeScale), edgeIndex


def CreateTorchGraphsTensor(data, global_features, global_scale, node_scale, edge_scale):
    """
    Function to loop through events and create graphs. This function skips networkx goes straight to tensor.

    Inputs:
      data (Pandas DataFrame): Row should contain an event with columns such as jet_pt, jet_phi etc. For objects with multiplicity, row entries should be lists containing an entry for each object of that type ordered by pt.
      global_features (list of strings): Global features to assign to graph.
      global_scale (numpy array): Scaling coefficients to scale global features by.
      node_scale (numpy array): Scaling coefficients to scale node features by.

    Returns:
      graphs (list): List of PyTorch Data objects containing graphs.
      df_booking (Pandas DataFrame): DataFrame of book-keeping information to identify graphs in future.
    """
    bookkeeping_features = [
        "eventNumber",
        "runNumber",
        "mcChannelNumber",
        "mH_label",
        "pseudo_mH",
        "nBTags_DL1r_70",
        "nJets",
    ]
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
        x, edge_attr, edge_index = event2Tensor(event, node_scale, edge_scale)
        keep_indices = [0, 2, 3, 4, 5]  # delete phi
        G = Data(
            x=x[:, keep_indices],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(event["IsSig"]).float(),
            w=torch.tensor(event["sample_weight"]).float(),
            u=torch.tensor(event[global_features]).float().view(-1, len(global_features)) / global_scale,
            pseudo_mH=torch.tensor(event["pseudo_mH"]).float() / 1000,
        )
        graphs.append(G)
        booking.append(event[bookkeeping_features])
    df_booking = pd.DataFrame(booking).reset_index(drop=True)
    print("\nDone")
    return graphs, df_booking


def getNodeFeatures(G):
    """
    Function to extract node features from graph and convert to torch tensor.

    Inputs:
      G (networks graph object): Must contain nodes with attribute named 'features'.
    Returns:
      x (torch tensor): Tensor of node features to be used to create Torch graph.

    """
    x = np.zeros(shape=(G.number_of_nodes(), len(G.nodes[0]["features"])))
    for node in G.nodes:
        x[int(node)] = np.asarray(G.nodes[node]["features"])
    x = torch.from_numpy(x)
    return x


def getEdgeFeatures(G):
    """
    Function to extract edge features from graph and convert to torch tensor.

    Inputs:
      G (networks graph object): Must contain edges with attribute named 'features'.
    Returns:
      e (torch tensor): Tensor of edge features to be used to create Torch graph.

    """
    e = np.zeros(shape=(G.number_of_edges(), len(G.edges[0, 1]["features"])))
    i = 0
    for edge in G.edges:
        e[i] = np.asarray(G.edges[edge]["features"])
        i = i + 1
    e = torch.from_numpy(e)
    return e


def getEdgeList(G):
    """
    Function to extract edge index from graph and convert to torch tensor.

    Inputs:
      G (networks graph object): Must contain edges.
    Returns:
      index (torch tensor): Tensor of edge index in format to be used to create Torch graph.

    """
    index = torch.from_numpy(np.asarray(G.edges())).t().contiguous().long()
    return index


class customDataset(Dataset):
    """
    PyTorch Dataset object overwritten for our purposes.

    Attributes:
          graphs (list): Containing PyTorch Data objects.
          booking (Pandas DataFrame): Book-keeping information.
          n (int): Number of files to split dataset into when saving/loading.
    """

    def __init__(self, graphs=[], booking=[]):
        super(customDataset, self).__init__()
        self.graphs = graphs
        self.booking = booking
        self.n = 10

    def save_to(self, path):
        """
        Function to save dataset to a path. Splits datasets and book-keeping dataframes into n files and saves.

        Inputs:
            path (string): Path where dataset should be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        remainder_n = len(self.booking) % self.n
        len_int = int(len(self.booking) - remainder_n)
        for i in range(self.n):
            if i == self.n - 1:
                end = len(self.booking)
            else:
                end = int((i * len_int + len_int) / self.n)
            start = int(i * len_int / self.n)
            self.booking.iloc[start:end].to_pickle("{}/booking_{}.pkl".format(path, i))
            with open("{}/graphs_{}.pkl".format(path, i), "wb") as f:
                pickle.dump(self.graphs[start:end], f)

    def download_from(self, path):
        """
        Function to load dataset from a path.

        Inputs:
            path (string): Path where dataset should be loaded from.
        """
        self.booking = pd.DataFrame()
        self.graphs = []
        for i in range(self.n):
            print("Downloading file {}/{}...".format(i + 1, self.n))
            self.booking = self.booking = pd.concat(
                [self.booking, pd.read_pickle("{}/booking_{}.pkl".format(path, i))], ignore_index=True
            )
            with open("{}/graphs_{}.pkl".format(path, i), "rb") as f:
                self.graphs = self.graphs + pickle.load(f)
        print("Done")

    def len(self):
        """
        Returns length of book-keeping i.e number of events.
        """
        return len(self.booking)

    def get(self, idx):
        """
        Returns graph from list of graphs with a given index.

        Inputs:
          idx (int): Index of graph in dataset.
        """
        return self.graphs[idx]

    def get_booking(self, idx):
        """
        Returns book-keeping info for a given index.

        Inputs:
          idx (int): Index of event in dataset
        """
        return self.booking.iloc[idx]


def editGraphs(dataset, global_keep_indices, node_keep_indices, edge_keep_indices):
    """
    Reduces graphs in dataset by removing global, node and edge features. User specifies indices of variables to keep and dataset is modified in-place.

    Inputs:
      dataset (customDataset object): Dataset to edit.
      global_keep_indices (list of ints): Indices of globals to keep in graphs.
      node_keep_indices (list of ints): Indices of node features to keep in graphs.
      edge_keep_indices (list of ints): Indices of edge features to keep in graphs.
    """
    for i in range(len(dataset)):
        data = dataset[i]
        data.u = data.u[:, global_keep_indices]
        data.x = data.x[:, node_keep_indices]
        data.edge_attr = data.edge_attr[:, edge_keep_indices]


def createSubgraphs(dataset, global_feats_before, global_feats_keep, node_feats_keep, edge_feats_keep):
    """
    Wrapper for editGraphs that allows user to specify names of features to keep. Initial node and edge feats are hard-coded. Naming convention removes object_ from prefix.

    Inputs:
      dataset (customDataset object): Dataset to edit.
      global_feats_before (list of strings): Names of global features in dataset.
      global_feats_keep (list of strings): Names of global features to keep.
      node_feats_keep (list of strings): Names of node features to keep.
      edge_feats_keep (list of strings): Names of edge features to keep.

    """
    global_keep_indices = [global_feats_before.index(feat) for feat in global_feats_keep]

    # node feats must be in this list
    node_features = ["pt", "phi", "eta", "e", "tagWeightBin_DL1r_Continuous", "qg_BDT_calibrated", "encoding"]
    node_keep_indices = [node_features.index(feat) for feat in node_feats_keep]

    # edge feats must be in this list
    edge_features = ["delta_phi", "delta_eta", "delta_R"]
    edge_keep_indices = [edge_features.index(feat) for feat in edge_feats_keep]

    for i in range(len(dataset)):
        data = dataset[i]
        data.u = data.u[:, global_keep_indices]
        data.x = data.x[:, node_keep_indices]
        data.edge_attr = data.edge_attr[:, edge_keep_indices]

    return dataset


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
