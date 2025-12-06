import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

from dymad.io import load_model
from dymad.models import KBF
from dymad.training import WeakFormTrainer
from dymad.utils import plot_summary, plot_multi_trajs, TrajectorySampler

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([
            [0., 1.],
            [-1., -0.1]])
def f(t, x, u):
    return (x @ A.T) + u
g = lambda t, x, u: x

config_gau = {
    "control" : {
        "kind": "gaussian",
        "params": {
            "mean": 0.5,
            "std":  1.0,
            "t1":   4.0,
            "dt":   0.2,
            "mode": "zoh"}}}

cases = [
    {"name": "kbf_cv",   "model" : KBF, "trainer": WeakFormTrainer, "config": 'lti_kbf_cv.yaml'},
]
IDX = [0]
labels = [cases[i]['name'] for i in IDX]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    iftrn = 1
    ifplt = 0
    ifprd = 0

    if iftrn:
        for _i in IDX:
            Model = cases[_i]['model']
            Trainer = cases[_i]['trainer']
            config_path = cases[_i]['config']

            trainer = Trainer(config_path, Model, max_workers=2)
            trainer.train()

    if ifplt:
        npz_files = [f'lti_{mdl}' for mdl in labels]
        npzs = plot_summary(npz_files, labels = labels, ifclose=False)
        for lbl, npz in zip(labels, npzs):
            print(f"Epoch time: {lbl} - {npz['avg_epoch_time']}")

    if ifprd:
        sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_gau)
        ts, xs, us, ys = sampler.sample(t_grid, batch=3)

        res = [xs]
        for _i in IDX:
            mdl, MDL = cases[_i]['name'], cases[_i]['model']
            _, prd_func = load_model(MDL, f'lti_{mdl}.pt')

            with torch.no_grad():
                _pred = prd_func(xs, ts, u=us)
            res.append(_pred)

        plot_multi_trajs(
            np.array(res), ts[0], "LTI",
            us=us, labels=['Truth']+labels, ifclose=False)

    plt.show()
