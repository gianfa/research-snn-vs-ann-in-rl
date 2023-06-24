# worked
dvc run --name data_load \
    --deps 'experiments/08-ESN-STDP-MLops/flows/ops01/data_loading.py' \
    --outs experiments/08-ESN-STDP-MLops/data/ds.pkl \
    /Users/giana/Library/Caches/pypoetry/virtualenvs/research-snn-vs-ann-in-rl-VunPnZrp-py3.9/bin/python \
        'experiments/08-ESN-STDP-MLops/flows/ops01/data_loading.py' \
        --config=experiments/08-ESN-STDP-MLops/params.yaml

dvc run --name training \
    /Users/giana/Library/Caches/pypoetry/virtualenvs/research-snn-vs-ann-in-rl-VunPnZrp-py3.9/bin/python \
        'experiments/08-ESN-STDP-MLops/flows/ops01/training.py' \
        --config=experiments/08-ESN-STDP-MLops/params.yaml

# not worked
dvc run --name training \
    --deps \
        'experiments/08-ESN-STDP-MLops/data/ds.pkl' \
        'experiments/08-ESN-STDP-MLops/flows/ops01/training.py' \
    --outs \
        experiments/08-ESN-STDP-MLops/data/perf_stats_before.pkl \
        experiments/08-ESN-STDP-MLops/data/perf_stats_after.pkl \
        experiments/08-ESN-STDP-MLops/data/perf_hist_stdp.pkl \
    /Users/giana/Library/Caches/pypoetry/virtualenvs/research-snn-vs-ann-in-rl-VunPnZrp-py3.9/bin/python \
        'experiments/08-ESN-STDP-MLops/flows/ops01/training.py' \
        --config=experiments/08-ESN-STDP-MLops/params.yaml

