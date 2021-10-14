import argparse
import os
import sys
from signal import SIGINT, signal

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import StandardScaler
from utils import binary_sampler, gen_data_nonlinear, handler

import miracle.logger as log
from miracle import MIRACLE

tf.disable_v2_behavior()
np.set_printoptions(suppress=True)
log.add(sink=sys.stderr, level="INFO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_sz", type=int, default=1000)
    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--reg_beta", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--ckpt_file", type=str, default="tmp.ckpt")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--missingness", type=float, default=0.2)
    args = parser.parse_args()
    signal(SIGINT, handler)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """
    Toy DAG
    The node '0' is the target in the Toy DAG
    """
    log.info("Doing TOY DAG")
    G = nx.DiGraph()
    for i in range(10):
        G.add_node(i)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 0)
    G.add_edge(3, 0)
    G.add_edge(3, 6)
    G.add_edge(3, 7)
    G.add_edge(6, 9)
    G.add_edge(0, 8)
    G.add_edge(0, 9)

    df = gen_data_nonlinear(G, SIZE=args.dataset_sz, sigmoid=False)

    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    X_MISSING = df.copy()  # This will have missing values
    X_TRUTH = df.copy()  # This will have no missing values for testing

    # Generate MCAR
    X_MASK = binary_sampler(
        1 - args.missingness, X_MISSING.shape[0], X_MISSING.shape[1]
    )
    X_MISSING[X_MASK == 0] = np.nan

    datasize = X_MISSING.shape[0]
    missingness = args.missingness
    feature_dims = X_MISSING.shape[1]

    log.info(
        f"""
        Datasize = {datasize}
        Missingness = {missingness}
        NumFeats =  {feature_dims}
        """
    )

    log.info("Baseline Mean Imputation")
    X = X_MISSING.copy()
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    imputed_data_x = X

    # Append indicator variables - One indicator per feature with missing values.
    missing_idxs = np.where(np.any(np.isnan(X_MISSING), axis=0))[0]
    df_mask = pd.DataFrame(X_MISSING)
    df_mask = df_mask.where(df_mask.isnull(), 0)
    df_mask = df_mask.mask(df_mask.isnull(), 1)
    indicators = df_mask[df_mask.columns[df_mask.isin([1]).any()]].values
    indicators = 1 - indicators
    X_MISSING_c = np.concatenate([X_MISSING, indicators], axis=1)
    imputed_data_x_c = np.concatenate([imputed_data_x, indicators], axis=1)
    num_input_missing = indicators.shape[1]
    X_MASK_c = np.concatenate(
        [X_MASK, np.ones((X_MASK.shape[0], num_input_missing))], axis=1
    )
    X_TRUTH_c = np.concatenate(
        [X_TRUTH, np.ones((X_TRUTH.shape[0], num_input_missing))], axis=1
    )

    # Initialize MIRACLE
    miracle = MIRACLE(
        num_train=X_MISSING_c.shape[0],
        num_inputs=X_MISSING_c.shape[1],
        reg_lambda=args.reg_lambda,
        reg_beta=args.reg_beta,
        n_hidden=32,
        w_threshold=0.3,
        ckpt_file=args.ckpt_file,
        missing_list=missing_idxs,
        reg_m=0.1,
        lr=0.0001,
        window=args.window,
        max_steps=args.max_steps,
    )

    # Train MIRACLE
    miracle_imputed_data_x, _pred_matrix, last_n = miracle.fit(
        X_MISSING_c,
        X_MASK_c,
        num_nodes=np.shape(X_MISSING_c)[1],
        X_seed=imputed_data_x_c,
    )

    log.info(f"Baseline RMSE {miracle.rmse_loss(X_TRUTH, imputed_data_x, X_MASK)}")
    log.info(
        f"MIRACLE RMSE {miracle.rmse_loss(X_TRUTH, miracle_imputed_data_x[:, : X_TRUTH.shape[1]], X_MASK)}"
    )
