class PairData(Data):
    """
    Class to store target and source graphs
    """

    def __init__(self, edge_index_s, edge_attr_s, x_s, u_s, pseudo_mH_s, edge_index_t, edge_attr_t, x_t, w):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.u_s = u_s
        self.pseudo_mH_s = pseudo_mH_s
        self.edge_attr_s = edge_attr_s
        self.w = w
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_attr_t = edge_attr_t

        self.node_w = w * torch.ones(size=self.x_t.size())
        self.edge_w = w * torch.ones(size=self.edge_attr_t.size())

    def __inc__(self, key, value):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)


def CreateGraphPairs(source_graphs, target_graphs):
    """
    Function to create PairData objects from list of target and source graphs
    """
    pairs = []
    for i in range(len(source_graphs)):

        graph_s = source_graphs[i]
        graph_t = target_graphs[i]

        pair = PairData(
            graph_s.edge_index,
            graph_s.edge_attr,
            graph_s.x,
            graph_s.u,
            graph_s.pseudo_mH,
            graph_t.edge_index,
            graph_t.edge_attr,
            graph_t.x,
            graph_s.w,
        )

        pairs.append(pair)
    return pairs


def train_edge_classifier(model, loader, cuda, criterion, optimizer):
    """
    Function to train one epoch of an edge classifier model. Loss is calculated between output graph edges and target graph edges.
    Todo: Implement sample weights

    Input:
    model (PyTorch model): must output an updated graph in form of node array and edge attribute array
    loader (PyTorch dataloader)
    cuda (PyTorch device)
    criterion (PyTorch loss function)
    optimizer (PyTorch optimiser)

    Returns:
    running loss (float): Running loss during training epoch
    """
    model.train()
    running_loss = 0
    for data in loader:  # Iterate in batches over the training dataset.
        # zero the parameter gradients
        data.to(cuda)
        optimizer.zero_grad()
        x_out, edge_attr_out = model(
            data.x_s, data.edge_index_s, data.edge_attr_s, data.u_s, data.pseudo_mH_s, data.x_s_batch
        )  # Perform a single forward pass.
        loss = criterion(edge_attr_out, data.edge_attr_t.float())  # Compute the loss per edge
        loss = loss.mean()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += loss.item()
    return running_loss
