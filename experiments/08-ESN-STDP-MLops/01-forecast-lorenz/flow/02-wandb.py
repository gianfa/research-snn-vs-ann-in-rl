""" Lorenz Forecast ESN + STDP: Performance vs SignalShift

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


"""
# %%
from config import *

from copy import deepcopy

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch

# from ops0701.config import *
from stdp.estimators import BaseESN
from stdp import funx as stdp_f

import src08.funx as src08_f


RUN_NAME = "exp-1"
RUN_NAME = f"{RUN_NAME}-{ek.funx.generate_current_datetime_string_no_sep()}"

TRIALS_DIR = EXP_DATA_DIR/"trails"
RUN_DIR = TRIALS_DIR/RUN_NAME
RUN_RESULTS_DIR = RUN_DIR/'produced_data'
for di in [TRIALS_DIR, RUN_DIR, RUN_RESULTS_DIR]:
    if not di.is_dir():
        di.mkdir()
report = ek.reporting.ReportMD(RUN_DIR/"README.md", title=EXP_NAME)
report.add_txt(f"Trial: {RUN_NAME}")

recompute = False

# %%

# if __name__ == '__main__':
#     params = argparse_config()

params = load_yaml(EXP_DIR/'sweeps.yaml')
ek.funx.yaml_save_dict(RUN_RESULTS_DIR/"params.pkl", params)
# print(params)
# report.add_code(txt=json.dumps(params, indent=4), language="python")

# %%Training
report.add_txt(f"## Training")

import wandb

wandb.login()


perf_diff = {"mae": [], "mse": []}
""" Performance differences
each value is about a `AFTER_STDP - NO_STDP` difference
"""

def wandb_main():
    wandb.init(project=f'{EXP_DIR.name}|{RUN_NAME}', entity='gianfa')
    config = wandb.config
    # report = ek.reporting.ReportMD(RUN_DIR/"README.md", title=EXP_NAME)

    # %%  --- Network Parameters ---
    bs = 1

    # ek.funx.yaml_save_dict(
    #     RUN_RESULTS_DIR/"params.pkl", config._items)
    # report.add_code(
    #     json.dumps(config._items,
    #     indent=4), language="python")

    reservoir_size = config['model_reservoir_size']
    STDP_scope = config['STDP_steps_scope']

    # STDP-Execute
    """ Many trials implementing STDP ESN weights update

    Trial steps
    -----------
    1. Initialize the ESN.
    2. Perform the STDP Update.

    STDP Update
    -----------
    1. Train the ESN and get the network state history from training
    2. Compute the new Reservoir weights, new_W, applying STDP to the states of
        the last 20 steps.
    3. Replace the Reservoir weights with the new ones just computed.

    """


    t0 = time.time()
    th = 0  # spike threshold
    perf_hist_nonstdp = {'mse': [], 'mae': [], 'r2': []}


    perf_hist_after_stdp = {'mse': [], 'mae': [], 'r2': []}
    """Mean performance history of STDP optimisation
    every element is the last performance along all the optimisation steps
    """

    perf_hist_stdp_inner: Dict[str, List] = {'mse': [], 'mae': [], 'r2': []}
    """Performance history of STDP optimisasion step
    every element is the performance at the specific optimisation step
    """

    perf_hist_stdp = []

    n_trials = config['exp_n_trials']
    n_STDP_steps = config['exp_n_STDP_steps']
    verbose = False

    # # Data Loading
    # report.add_txt(
    #     f"### Data Loading\nLorenz oscillator is generated with the "
    #     + r"following parameters: $\sigma=12; \rho=30; \beta=2.7$.")
    X_train, X_valid, X_test, y_train, y_valid, y_test = \
        src08_f.expt_generate_new_lorenz_data(
            example_len = config['data_example_len'],
            test_size = config['data_test_size'],
            valid_size = config['data_valid_size'],
            recompute = True,
            ds_path = RUN_RESULTS_DIR/f"ds_lorenz.pkl",
            shift = config['data_shift'],  # forecasted delay
            s=12, r=30, b=2.700,
            time_last = True,
        )

    W_hist_nonstdp = []
    W_hist_stdp =  []

    # report.add_txt(
    # f"### Data Loading\nLorenz oscillator is generated with the "
    # + "following parameters: $\sigma=12; \rho=30; \beta=2.7$."
    # + "A series of trials is initiated. This is to have at the end a "
    # + "statistical analysis of the expected performance, considering "
    # + "the aspect of uncertainty given by the inherent chaotic nature.")
    for i in range(n_trials):
    
        print(f"INFO: Trial #{i}")

        esn = BaseESN(
            input_size = X_train.shape[0],
            reservoir_size=reservoir_size,
            output_size = y_train.shape[0],
            connections = (stdp_f.generate_simple_circle_connections_mask(
                (reservoir_size, reservoir_size)) > 0).int(),
        )

        # Non-STDP
        state_hist = esn.train(X_train.float(), y_train.float())
        y_out = esn.predict(X_valid.float())
        y_pred = y_out
        
        report = src08_f.evaluate_regression_report(y_valid, y_pred)
        for m_name, m_val in report.items():
            if m_name not in perf_hist_nonstdp:
                perf_hist_nonstdp[m_name] = []
            perf_hist_nonstdp[m_name].append(m_val)
            wandb.log({f"perf_hist_nonstdp{m_name}": m_val})

        
        W_hist_nonstdp.append(esn.W.data)
        wandb.log({f"W_hist_nonstdp_avg": esn.W.data.mean()})
        
        # STDP Update
        perf_hist_stdp_inner = {}
        W_hist_stdp.append([])
        for epoch in range(n_STDP_steps):
            t_i = time.time()

            # STDP
            raster = (state_hist > th).to(int)[:, -STDP_scope:]
            # update hidden connections
            reservoir_weights = esn.W.clone()
            reservoir_connections = esn.connections

            # print(layer.weight.mean())
            new_W = stdp_f.stdp_step(
                reservoir_weights,
                connections=reservoir_connections,
                raster=raster,
                spike_collection_rule = stdp_f.all_to_all,
                dw_rule = "sum",
                max_delta_t=4,
                STDP_kwargs={
                    'A_plus': config['STDP_A_plus'],
                    'A_minus': config['STDP_A_minus'],
                    'tau_plus': config['STDP_tau_plus'],
                    'tau_minus': config['STDP_tau_minus']
                }
            )
            # Normalize weights
            # new_W /= new_W.max()
            new_W = ((new_W / new_W.abs().max()) * 2 - 1)
            if epoch % 2 == 0:
                new_W = (0.5 * new_W + \
                    torch.randn_like(esn.W) * 0.5)
            new_W *= esn.connections
            # ensure weights matrix reflects the connections
            n_exceding_connections = (
                (new_W != 0).to(torch.int) - \
                    (reservoir_connections != 0).to(torch.int)
                ).sum()
            assert n_exceding_connections == 0, f"{n_exceding_connections}"
            
            # Update ESN weights
            esn.W = new_W
            assert esn.W.equal(new_W)
            W_hist_stdp[-1].append(esn.W.data)

            if verbose: print(f"STDP executed in {time.time() - t_i:.0f}s")

            # # Retrain after STDP and evaluate
            state_hist = esn.train(X_train.float(), y_train.float())
            y_out = esn.predict(X_valid.float())
            y_pred = y_out
            
            report = src08_f.evaluate_regression_report(y_valid, y_pred)
            for m_name, m_val in report.items():
                if m_name not in perf_hist_stdp_inner:
                    perf_hist_stdp_inner[m_name] = []
                perf_hist_stdp_inner[m_name].append(m_val)
        
        wandb.log({f"W_hist_stdp_avg": esn.W.data.mean()})

        # perf_hist_stdp.append(
        #     {k: np.mean(v) for k, v in perf_hist_stdp_inner.items()})
        perf_hist_stdp_inner = {
            k: torch.Tensor(v) for k, v in perf_hist_stdp_inner.items()}
        perf_hist_stdp.append(perf_hist_stdp_inner)

        for m_name, m_hist in perf_hist_stdp_inner.items():
            if m_name not in perf_hist_stdp_inner:
                perf_hist_after_stdp[m_name] = []
            perf_hist_after_stdp[m_name].append(m_hist[-1].item())
            wandb.log({f"perf_hist_after_stdp_{m_name}": m_hist[-1].item()})
        
        perf_hist_diff = {}
        for m_name in report.keys():
            perf_hist_diff[m_name] = \
                perf_hist_nonstdp[m_name][-1] - perf_hist_after_stdp[m_name][-1]
            wandb.log({
                f"perf_hist_{m_name}-diff": perf_hist_diff[m_name]})

        perf_diff['mae'].append(perf_hist_diff['mae'])
        perf_diff['mse'].append(perf_hist_diff['mse'])
        wandb.log({
            'diff_mae': perf_hist_diff['mae'],
            'diff_mse': perf_hist_diff['mse']
        })

        print(f"t: {time.time() - t0:.0f}s")

    print(f"STDP performance stats:")
    perf_stats_before = pd.DataFrame(perf_hist_nonstdp).describe()
    print(perf_stats_before)


    perf_stats_after = pd.DataFrame(perf_hist_after_stdp).describe()
    print(perf_stats_after)
    
    W_hist_stdp = [
        torch.stack(W_hist_stdp_i) for W_hist_stdp_i in W_hist_stdp]
    W_hist_stdp = torch.stack(W_hist_stdp)
    W_hist_nonstdp = torch.stack(W_hist_nonstdp)

    ek.funx.pickle_save_dict(
        RUN_RESULTS_DIR/'perf_hist_nonstdp.pkl', perf_hist_nonstdp)
    ek.funx.pickle_save_dict(
        RUN_RESULTS_DIR/'perf_hist_after_stdp.pkl', perf_hist_after_stdp)
    ek.funx.pickle_save_dict(
        RUN_RESULTS_DIR/'perf_hist_stdp.pkl',
        {'perf_hist_stdp': perf_hist_stdp})
    ek.funx.pickle_save_dict(
        RUN_RESULTS_DIR/'W_hist_stdp.pkl', {'W_hist_stdp': W_hist_stdp})
    ek.funx.pickle_save_dict(
        RUN_RESULTS_DIR/'W_hist_nonstdp.pkl', {'W_hist_nonstdp': W_hist_nonstdp})


sweep_id = wandb.sweep(
    sweep=params, project=f'{EXP_DIR.name}|{RUN_NAME}')

wandb.agent(sweep_id, function=wandb_main)

# %%
