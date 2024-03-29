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
import itertools

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch

# from ops0701.config import *
from stdp.estimators import BaseESN
from stdp import funx as stdp_f

import src08.funx as src08_f


TRIAL_NAME = f"exp-1-{ek.funx.generate_current_datetime_string_no_sep()}"

TRIALS_DIR = EXP_DATA_DIR/"trails"
SUBEXP_DIR = TRIALS_DIR/TRIAL_NAME
SUBEXP_RESULTS_DIR = SUBEXP_DIR/'produced_data'
for di in [TRIALS_DIR, SUBEXP_DIR, SUBEXP_RESULTS_DIR]:
    if not di.is_dir():
        di.mkdir()
report = ek.reporting.ReportMD(SUBEXP_DIR/"README.md", title=EXP_NAME)
report.add_txt(f"Trial: {TRIAL_NAME}")

recompute = False

# %%

# if __name__ == '__main__':
#     params = argparse_config()

params = load_yaml(EXP_DIR/'params.yaml')
print(params)

# %% Training

grid = {
    'example_len': [100, 1000, 10000],
    'shift': range(1, 20, 2),
    'n_STDP_steps': range(2, 10, 2),
    'STDP_scope': [10, 20, 30],
    'reservoir_size': [10, 50, 100, 200, 500, 1000],
    'decay': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    # 'A_plus': [0.1, 0.2, 0.5, 0.7],
    # 'A_minus': [0.1, 0.2, 0.5, 0.7],
    # 'tau_plus': np.logspace(-3, -1.5, 5),
    # 'tau_minus': np.logspace(-3, -1.5, 5),
}

grid_tuples = list(itertools.product(*grid.values()))
print(f"INFO: {len(grid_tuples)} tuples in grid")

# %%

grid_names = list(grid.keys())
for exp_i, exp_pi in enumerate(grid_tuples):
    
    exp_dict = {ni: vi for ni, vi in zip(grid_names, exp_pi)}
    # prepare folder
    exp_name = f"{exp_pi}"
    exp_dir = SUBEXP_RESULTS_DIR/exp_name
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    else:
        continue

    # update params
    params['data']['example_len'] = exp_dict['example_len']
    params['data']['shift'] = exp_dict['shift']
    params['experiment']['n_STDP_steps'] = exp_dict['n_STDP_steps']
    params['experiment']['STDP_scope'] = exp_dict['STDP_scope']
    params['model']['decay'] = exp_dict['decay']
    # params['STDP']['A_plus'] = exp_dict['A_plus']
    # params['STDP']['A_minus'] = exp_dict['A_minus']
    # params['STDP']['tau_plus'] = exp_dict['tau_plus']
    # params['STDP']['tau_minus'] = exp_dict['tau_minus']
        
    ek.funx.pickle_save_dict(exp_dir/"params.pkl", params)
    print(params)


    reservoir_size = params['model']['reservoir_size']
    STDP_scope = params['experiment']['STDP_scope']

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

    n_trials = params['experiment']['n_trials']
    n_STDP_steps = params['experiment']['n_STDP_steps']
    verbose = False

    # Data Loading
    X_train, X_valid, X_test, y_train, y_valid, y_test = \
        src08_f.expt_generate_new_lorenz_data(
            example_len = params['data']['example_len'],
            test_size = params['data']['test_size'],
            valid_size = params['data']['valid_size'],
            recompute = True,
            ds_path = exp_dir/f"ds_lorenz.pkl",
            shift = params['data']['shift'],  # forecasted delay
            s=12, r=30, b=2.700,
            time_last = True,
        )

    W_hist_nonstdp = []
    W_hist_stdp =  []

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

        
        W_hist_nonstdp.append(esn.W.data)
        
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
                    'A_plus': params['STDP']['A_plus'],
                    'A_minus': params['STDP']['A_minus'],
                    'tau_plus': params['STDP']['tau_plus'],
                    'tau_minus': params['STDP']['tau_minus']
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
            
        # perf_hist_stdp.append(
        #     {k: np.mean(v) for k, v in perf_hist_stdp_inner.items()})
        perf_hist_stdp_inner = {
            k: torch.Tensor(v) for k, v in perf_hist_stdp_inner.items()}
        perf_hist_stdp.append(perf_hist_stdp_inner)

        for m_name, m_hist in perf_hist_stdp_inner.items():
                if m_name not in perf_hist_stdp_inner:
                    perf_hist_after_stdp[m_name] = []
                perf_hist_after_stdp[m_name].append(m_hist[-1].item())

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

    ek.funx.pickle_save_dict(exp_dir/'perf_hist_nonstdp.pkl', perf_hist_nonstdp)
    ek.funx.pickle_save_dict(exp_dir/'perf_hist_after_stdp.pkl', perf_hist_after_stdp)
    ek.funx.pickle_save_dict(
        exp_dir/'perf_hist_stdp.pkl',
        {'perf_hist_stdp': perf_hist_stdp})
    ek.funx.pickle_save_dict(
        exp_dir/'W_hist_stdp.pkl', {'W_hist_stdp': W_hist_stdp})
    ek.funx.pickle_save_dict(
        exp_dir/'W_hist_nonstdp.pkl', {'W_hist_nonstdp': W_hist_nonstdp})

# %%
