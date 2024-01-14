# %%
""" Lorenz Forecast ESN + STDP: Performance vs SignalShift
[Good for experiments with wandb]

Q: Is it possible to achieve an improvement in the classification
    performance of an ESN by applying STDP?


-------------------------

Exp Observations
----------------
1. It is prone to exploding weights since STDP will amplify indefinitely.
    -> Normalization before update.
2. 

   
References
----------
1. SCHAETTI, Nils; SALOMON, Michel; COUTURIER, Raphaël. Echo state networks-based reservoir computing for mnist handwritten digits recognition. In: 2016 IEEE Intl Conference on Computational Science and Engineering (CSE) and IEEE Intl Conference on Embedded and Ubiquitous Computing (EUC) and 15th Intl Symposium on Distributed Computing and Applications for Business Engineering (DCABES). IEEE, 2016. p. 484-491.

TODOs
-----
* v differenza stdp non stdp 
in funzione della lunghezza del segnale
* distribuzione dei pesi nei confronti delle performance


# python -m 03-Forecast-Lorenz-perf_wrt_signal_shift_wandb.py
"""
# %%
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb

sys.path += ["..", "../..", "../../.."]
from experimentkit_in.visualization import get_cmap_colors
from experimentkit_in.funx import pickle_save_dict, pickle_load, load_yaml
from experimentkit_in.metricsreport import MetricsReport
from experimentkit_in.generators.time_series import gen_lorenz
from stdp.estimators import BaseESN
from stdp import funx as stdp_f

import src07.funx as src07_f

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


ROOT = Path("../../../")
# ROOT = Path("../../")
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "experiments/07-ESN-STDP"
EXP_DATA_DIR = EXP_DIR / "data"
EXP_REPORT_DIR = EXP_DIR / "report"
EXP_NAME = "exp7-Duffing"
EXP_RESULTS_DIR = EXP_DATA_DIR / EXP_NAME / "produced_data"

WANDB_PROJECT_NAME = f'STDP-ESN-{EXP_NAME}-2'

assert ROOT.resolve().name == 'research-snn-vs-ann-in-rl'
assert EXP_DIR.exists(), f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"

recompute = False
use_wandb = False
report_path = EXP_REPORT_DIR/(str(EXP_NAME) + ".md")

src07_f.report_md_append(f"# {WANDB_PROJECT_NAME}", report_path)

# %%

wandb.login()

sweep_configuration = load_yaml("../sweep-bayesoptim.yaml")
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project=WANDB_PROJECT_NAME)

txt = """
\n\nThe hyperparameters tuning has been performed by
Bayesian Optimisation  \n
"""
src07_f.report_md_append(txt, report_path)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# %% Training

# params_ = {
#     'example_len': 100,
#     'shift': 1,
#     'reservoir_size': 5,
#     'n_STDP_steps': 2,
#     'STDP_scope': 10,
#     'n_trials': 10
#     }

exp_id = "exp_len_shift_STDPsteps_scope"

# %%


# Initialize a new wandb run
def wandb_main():
    run = wandb.init(project=WANDB_PROJECT_NAME, entity='gianfa')

    params = wandb.config
    logger.info(f"file: {__file__}")
    logger.info(f"params: {params}")

    # prepare folder
    folder_name = "_".join([
        str(params.example_len),
        str(params.shift),
        str(params.reservoir_size),
        str(params.n_STDP_steps),
        str(params.STDP_scope),
        str(params.n_trials)
    ])
    # rename wandb single sweep
    run.name = folder_name

    reservoir_size = params.reservoir_size
    STDP_scope = params.STDP_scope

    logger.info(f"folder_name: {folder_name}")
    exp_dir = EXP_RESULTS_DIR / folder_name
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    # STDP-Execute
    """ Many trials implementing STDP ESN weights update

    Trial steps
    -----------
    1. Initialize the ESN.
    2. Perform the STDP Update.

    STDP Update
    -----------
    1. Train the ESN and get the network state history from training
    2. Compute the new Reservoir weights, new_W, applying STDP to the
        states of the last 20 steps.
    3. Replace the Reservoir weights with the new ones just computed.

    """

    params_dict = wandb.config.as_dict()
    pickle_save_dict(exp_dir / "params.pkl", params_dict)

    # %%
    t0 = time.time()
    th = 0  # spike threshold
    perf_hist_nonstdp = {"mse": [], "mae": [], "r2": []}

    perf_hist_after_stdp = {"mse": [], "mae": [], "r2": []}
    """Mean performance history of STDP optimisation
    every element is the last performance along all the optimisation steps
    """

    perf_hist_stdp_inner: Dict[str, List] = {
        "mse": [], "mae": [], "r2": []}
    """Performance history of STDP optimisasion step
    every element is the performance at the specific optimisation step
    """

    perf_hist_stdp = []

    n_trials = params.n_trials
    n_STDP_steps = params.n_STDP_steps
    verbose = False

    # Data Loading
    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = src07_f.expt_generate_new_duffing_data(
            example_len=params.example_len,
            test_size=0.2,
            valid_size=0.15,
            recompute=True,
            ds_path=exp_dir / "ds_duffing.pkl",
            shift=params.shift,  # forecasted dey
            alpha=1, beta=-1, gamma=0.5, delta=0.3, omega=1,
        )
    logger.info(f"X_train.shape: {X_train.shape}")

# %%

    W_hist_nonstdp = []
    W_hist_stdp = []

    for i in range(n_trials):

        logger.info(f"INFO: Trial #{i}")

        esn = BaseESN(
            input_size=X_train.shape[0],
            reservoir_size=reservoir_size,
            output_size=y_train.shape[0],
            connections=(
                stdp_f.generate_simple_circle_connections_mask(
                    (reservoir_size, reservoir_size)
                )
                > 0
            ).int(),
        )

        # Non-STDP
        state_hist = esn.train(X_train.float(), y_train.float())
        y_out = esn.predict(X_valid.float())
        y_pred = y_out

        report = src07_f.evaluate_regression_report(y_valid, y_pred)
        for m_name, m_val in report.items():
            if m_name not in perf_hist_nonstdp:
                perf_hist_nonstdp[m_name] = []
            perf_hist_nonstdp[m_name].append(m_val)

        W_hist_nonstdp.append(esn.W.data)

        # STDP Update
        perf_hist_stdp_inner = {}
        W_hist_stdp.append([])

        wandb.log({
            'trial': i,
            'mae': perf_hist_nonstdp['mae'][-1],
            'r2': perf_hist_nonstdp['r2'][-1],
            'weights_avg': W_hist_nonstdp[-1].mean(),
        })
        for epoch in range(n_STDP_steps):
            t_i = time.time()

            # STDP
            raster = (state_hist > th).to(int)[:, -STDP_scope:]
            # update hidden connections
            reservoir_weights = esn.W.clone()
            reservoir_connections = esn.connections

            new_W = stdp_f.stdp_step(
                reservoir_weights,
                connections=reservoir_connections,
                raster=raster,
                spike_collection_rule=stdp_f.all_to_all,
                dw_rule="sum",
                max_delta_t=4,
            )
            # Normalize weights
            # new_W /= new_W.max()
            new_W = (new_W / new_W.abs().max()) * 2 - 1
            if epoch % 2 == 0:
                new_W = 0.5 * new_W + torch.randn_like(esn.W) * 0.5
            new_W *= esn.connections

            # ensure weights matrix reflects the connections
            n_exceding_connections = (
                (new_W != 0).to(torch.int)
                - (reservoir_connections != 0).to(torch.int)
            ).sum()
            assert n_exceding_connections == 0, f"{n_exceding_connections}"

            # Update ESN weights
            esn.W = new_W
            assert esn.W.equal(new_W)
            W_hist_stdp[-1].append(esn.W.data)

            if verbose:
                logger.info(f"STDP executed in {time.time() - t_i:.0f}s")

            # # Retrain after STDP and evaluate
            state_hist = esn.train(X_train.float(), y_train.float())
            y_out = esn.predict(X_valid.float())
            y_pred = y_out

            report = src07_f.evaluate_regression_report(y_valid, y_pred)
            for m_name, m_val in report.items():
                if m_name not in perf_hist_stdp_inner:
                    perf_hist_stdp_inner[m_name] = []
                perf_hist_stdp_inner[m_name].append(m_val)

            with open("todel-mae", "w+") as f:
                f.write(str(type(esn.W.data)))

            delta_MAE = (perf_hist_nonstdp['mae'][-1]
                - perf_hist_stdp_inner['mae'][-1])
            wandb.log({
                'STDP_MAE': perf_hist_stdp_inner['mae'][-1],
                'STDP_weights_avg': np.mean(W_hist_stdp[-1]),
                'no-STDP-MAE - STDP-MAE':  delta_MAE,
            })


        # perf_hist_stdp.append(
        #     {k: np.mean(v) for k, v in perf_hist_stdp_inner.items()})
        perf_hist_stdp_inner = {
            k: torch.Tensor(v) for k, v in perf_hist_stdp_inner.items()
        }
        perf_hist_stdp.append(perf_hist_stdp_inner)

        for m_name, m_hist in perf_hist_stdp_inner.items():
            if m_name not in perf_hist_stdp_inner:
                perf_hist_after_stdp[m_name] = []
            perf_hist_after_stdp[m_name].append(m_hist[-1].item())

        logger.info(f"t: {time.time() - t0:.0f}s")

        logger.info("STDP performance stats:")
        perf_stats_before = pd.DataFrame(perf_hist_nonstdp).describe()
        logger.info(perf_stats_before)

        perf_stats_after = pd.DataFrame(perf_hist_after_stdp).describe()
        logger.info(perf_stats_after)

        W_hist_stdp_tens = [
            torch.stack(W_hist_stdp_i) for W_hist_stdp_i in W_hist_stdp]
        # W_hist_stdp_tens = torch.stack(W_hist_stdp)

        pickle_save_dict(exp_dir / "perf_hist_nonstdp.pkl", perf_hist_nonstdp)
        pickle_save_dict(
            exp_dir / "perf_hist_after_stdp.pkl", perf_hist_after_stdp)
        pickle_save_dict(
            exp_dir / "perf_hist_stdp.pkl", {"perf_hist_stdp": perf_hist_stdp}
        )
        pickle_save_dict(
            exp_dir / "W_hist_stdp.pkl", {"W_hist_stdp": W_hist_stdp_tens})
        pickle_save_dict(
            exp_dir / "W_hist_nonstdp.pkl", {"W_hist_nonstdp": torch.stack(W_hist_nonstdp)}
        )


wandb.agent(sweep_id, function=wandb_main)

logger.info("Done")

# %%