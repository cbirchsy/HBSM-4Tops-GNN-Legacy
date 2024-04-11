
These are demo notebooks for running the HBSM 4-Top GNN training.

The code is quite specific and hard-coded to this analysis so will require modification to work for any other use case. Details of root file preprocessing is not be provided.

The general structure is:
1) Convert ROOT files to pickled dataframes, filter events and add columns such as sig/bkg label and signal mass label
2) Convert dataframes to PyG graphs, many features related to input are hard-coded here fyi
3) Run training! Done

