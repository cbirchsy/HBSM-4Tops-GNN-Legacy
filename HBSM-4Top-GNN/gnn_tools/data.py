from __future__ import print_function
import time
import numpy as np
import pandas as pd
import pickle
import os


def initialise_weights(df):
    """
    Takes in DataFrame of events and adds a initial weight column. Weights are initially calculated as the product of normalisation, MC, pileup, lepton SF, jvt and btag SF weights. These are then scaled by a luminosity corresponding to runNumber. Do this before applying reweighting factors.

        Inputs:
          df (Pandas DataFrame): DataFrame containing events as rows. Must contain columns explicitly named 'weight_normalise', 'weight_mc', 'weight_pileup', 'weight_leptonSF', 'weight_jvt', 'weight_bTagSF_DL1r_Continuous' and 'runNumber'. (this is hardcoded, modify function if your weights are named differently )

        Returns:
          df (Pandas DataFrame): Updated dataframe with a new column named 'weight_rw'.
    """
    df["weight_rw"] = (
        df.weight_normalise
        * df.weight_mc
        * df.weight_pileup
        * df.weight_leptonSF
        * df.weight_jvt
        * df.weight_bTagSF_DL1r_Continuous
    )
    df.loc[df["runNumber"] == 284500, "weight_rw"] = df.loc[df["runNumber"] == 284500, "weight_rw"] * (
        3219.56 + 32988.1
    )
    df.loc[df["runNumber"] == 300000, "weight_rw"] = df.loc[df["runNumber"] == 300000, "weight_rw"] * 44307.4
    df.loc[df["runNumber"] == 310000, "weight_rw"] = df.loc[df["runNumber"] == 310000, "weight_rw"] * 58450.1
    return df


def bkg_reweighting(df):
    """
    Takes in DataFrame of background events and applies ttbar background reweighting based on NN reweighting factors. HBSM 4-top analysis specific code dependent on NN reweighting factors from https://gitlab.cern.ch/atlas-phys/exot/hqt/bsm_h_4t/tmva_tool. NN reweighting implementation are different in 1L and 2L. HF scaling is ((HF_SimpleClassification==-1)*1.30+(HF_SimpleClassification==0&&Sum$(jet_truthflav==5)<=2)*0.86+(HF_SimpleClassification==1||(HF_SimpleClassification==0&&Sum$(jet_truthflav==5)>2))*1.18).

        Inputs:
          df (Pandas DataFrame): DataFrame containing background only events as rows. Must contain columns 'score_hfinckin', 'weight_rw', 'HF_SimpleClassification', 'jet_truthflav'.
          channel (string): '1L' or '2L'

        Returns:
          df (Pandas DataFrame): Updated dataframe with reweighting factors applied to 'weight_rw' column.
    """
    # NN reweighting
    df["weight_rw"] = df["weight_rw"] * df["nnRewFactor"]

    A = 1.60
    C = 0.84
    B = 1.21

    # HF factor scaling
    df.loc[df["HF_SimpleClassification"] == -1, "weight_rw"] = (
        df.loc[df["HF_SimpleClassification"] == -1, "weight_rw"] * A
    )  # 1.30
    df.loc[df["HF_SimpleClassification"] == 1, "weight_rw"] = (
        df.loc[df["HF_SimpleClassification"] == 1, "weight_rw"] * B
    )  # 1.18
    df["jet_truthflav5_sum"] = [
        list(entry).count(5) for entry in df["jet_truthflav"]
    ]  # new column of Sum$(jet_truthflav==5)
    df.loc[(df["HF_SimpleClassification"] == 0) & (df["jet_truthflav5_sum"] > 2), "weight_rw"] = (
        df.loc[(df["HF_SimpleClassification"] == 0) & (df["jet_truthflav5_sum"] > 2), "weight_rw"] * B
    )  # 1.18
    df.loc[(df["HF_SimpleClassification"] == 0) & (df["jet_truthflav5_sum"] <= 2), "weight_rw"] = (
        df.loc[(df["HF_SimpleClassification"] == 0) & (df["jet_truthflav5_sum"] <= 2), "weight_rw"] * C
    )  # 0.86

    return df


def apply_reweighting(df):
    """
    Takes in DataFrame of events and adds 'weight_rw' column containing initial weights with 'initialise_weights' function. Splits signal and background based on 'IsSig' column and applies bkg NN reweighting via 'bkg_reweighting' function. HBSM 4-top analysis specific code, see 'bkg_reweighting'.

        Inputs:
          df (Pandas DataFrame): DataFrame containing events as rows. Must contain 'IsSig' column. Also see requirements of 'initialise_weights' and 'bkg_reweighting' function.
          channel (string): '1L' or '2L'

        Returns:
          df (Pandas DataFrame): Updated dataframe with a new column named 'weight_rw' containing calculated weights with reweighting applied for bkg events.
    """
    df = initialise_weights(df)
    df_bkg = df.query("IsSig==0")
    df_sig = df.query("IsSig==1")
    df_bkg = bkg_reweighting(df_bkg)
    df = df_bkg.append(df_sig, ignore_index=True)
    df = df.reset_index(drop=True)
    return df


def generate_uniform_pseudomass(df):
    """
    Takes in DataFrame of events and adds 'pseudo_mH' column which applies a uniform pseudo mass value to background events between 400 and 1000 GeV. For signal event column is filled with real mH value. HBSM 4-top analysis specific code.

        Inputs:
          df (Pandas DataFrame): DataFrame containing events as rows. Must contain 'IsSig' and 'mH_label' columns.

        Returns:
          df (Pandas DataFrame): Updated dataframe with 'pseudo_mH' column.
    """
    df["pseudo_mH"] = df["mH_label"]
    df.loc[df["IsSig"] == 0, "pseudo_mH"] = np.random.randint(4, 11, (df.query("IsSig==0")).shape[0]) * 100
    return df


def balance_feature(df, feature):
    """
    Takes in DataFrame of events and generate 'sample_weight' column. Sample weights are calculated by balancing 'weights_rw' between for a given feature. Such that sum(weights) is same for every feature category e.g. pseudo mass

        Inputs:
          df (Pandas DataFrame): DataFrame containing events as rows. Must contain 'weight_rw' column.
          feature (string): Column name of feature to balance.

        Returns:
          df (Pandas DataFrame): Updated dataframe with 'sample_weights' column added.
    """
    df["sample_weight"] = df["weight_rw"]
    occurances = df[feature].value_counts()
    # divide weights within category by average weight of all events in each category
    for occ in occurances.keys():
        sumWeights = sum(df.query("{}=={}".format(feature, occ))["weight_rw"].to_numpy())
        df.loc[df[feature] == occ, "sample_weight"] = df.loc[df[feature] == occ, "sample_weight"] * 1 / sumWeights
    df["sample_weight"] = df["sample_weight"] * 1 / np.mean(df["sample_weight"].to_numpy())
    return df


def add_sample_weights(df):
    """
    Create sample weights by generating pseudo mass values and re-balancing weights such that Sum(weights) is same in each pseudo mass category and Sum(signal weights)=Sum(background weights) in each. HBSM 4-top analysis specific code.

        Inputs:
          df (Pandas DataFrame): DataFrame containing events as rows.

        Returns:
          df (Pandas DataFrame): Updated dataframe with 'sample_weights' column added.
    """
    df = generate_uniform_pseudomass(df)
    # balance features so masses are balanced
    df = balance_feature(df, "mH_label")
    # balance sig/bkg frac by multiplying bkg weight by number of sig classes
    df.loc[df["IsSig"] == 0, "sample_weight"] = df.loc[df["IsSig"] == 0, "sample_weight"] * 7
    return df


def add_HT_ratio(df):
    """
    Add HT ratio variable to DataFrame of events. HT ratio is defined in HBSM 4-top analysis as sum of first 4 leading jet pTs divided by the sum of pTs of the remaining jets.

        Inputs:
          df (Pandas DataFrame): DataFrame containing events as rows.

        Returns:
          df (Pandas DataFrame): Updated dataframe with 'HT_ratio' column added.
    """
    ht_ratio = []
    for index, event in df.iterrows():
        ht_ratio.append(sum(np.asarray(event["jet_pt"])[0:3]) / sum(np.asarray(event["jet_pt"])[4:-1]))
    df["HT_ratio"] = ht_ratio
    return df


def add_sum_rcjets_d12_d23(df):
    """
    Add Sum_rcjet_d12 and Sum_rcjet_d23 variables to DataFrame of events.

        Inputs:
          df (Pandas DataFrame): DataFrame containing events as rows.

        Returns:
          df (Pandas DataFrame): Updated dataframe with 'Sum_rcjet_d12' and 'Sum_rcjet_d23' columns added.
    """
    sum_d12 = []
    sum_d23 = []
    for index, event in df.iterrows():
        sum_d12.append(sum(event["rcjet_d12"]))
        sum_d23.append(sum(event["rcjet_d23"]))
    df["Sum_rcjet_d12"] = sum_d12
    df["Sum_rcjet_d23"] = sum_d23
    return df
