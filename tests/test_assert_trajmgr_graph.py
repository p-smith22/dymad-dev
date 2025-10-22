import copy
import numpy as np
import torch

from dymad.io import DynData, TrajectoryManagerGraph
from dymad.transform import make_transform
from dymad.utils import adj_to_edge

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s.t, _t.t), f"{label} failed: {_s.t} != {_t.t}"
        assert np.allclose(_s.x, _t.x), f"{label} failed: {_s.x} != {_t.x}"
        assert _s.y is None
        assert _t.y is None
        assert np.allclose(_s.u, _t.u), f"{label} failed: {_s.u} != {_t.u}"
        assert np.allclose(_s.p, _t.p), f"{label} failed: {_s.p} != {_t.p}"
        for _s_ei, _t_ei in zip(_s.ei, _t.ei):
            assert np.allclose(_s_ei, _t_ei), f"{label} failed: {_s_ei} != {_t_ei}"
        for _s_ew, _t_ew in zip(_s.ew, _t.ew):
            assert np.allclose(_s_ew, _t_ew), f"{label} failed: {_s_ew} != {_t_ew}"
        assert _s.ea is None
        assert _t.ea is None
    print(f"{label} passed.")

def test_dyndata_graph(ltg_data):
    data = np.load(ltg_data, allow_pickle=True)
    ts = torch.tensor(data['t'][:2, :4])
    xs = torch.tensor(data['x'][:2, :4])
    us = torch.tensor(data['u'][:2, :4])
    ps = torch.tensor(data['p'][:2])

    e1 = torch.tensor([[0,1,2],[1,0,0]]).transpose(0,1)
    w1 = torch.tensor([1.0, 1.0, 1.0])
    e2 = torch.tensor([[0,2],[1,0]])
    w2 = torch.tensor([1.0, 3.0])
    es = [[e1, e2, e2, e1],
          [e2, e1, e2, e1]]
    ws = [[w1, w2, w2, w1],
          [w2, w1, w2, w1]]

    Dlist = [
        DynData(t=t, x=x, u=u, p=p, ei=e, ew=w)
        for t, x, u, p, e, w in zip(ts, xs, us, ps, es, ws)
    ]
    data = DynData.collate(Dlist)
    dref = DynData(t=ts, x=xs, u=us, p=ps, ei=es, ew=ws)
    check_data([data], [dref], label='DynData graph collate')

    N = 3
    trun = data.truncate(N)
    dref = DynData(
        t=ts[:,:N], x=xs[:,:N], u=us[:,:N], p=ps,
        ei=[_e[:N] for _e in es], ew=[_w[:N] for _w in ws])
    check_data([trun], [dref], label='DynData truncate')

    W, S = 3, 1
    ufld = data.unfold(W, S)
    dref = DynData(
        t = torch.vstack([ts[:,:3], ts[:,1:]]),
        x = torch.vstack([xs[:,:3], xs[:,1:]]),
        u = torch.vstack([us[:,:3], us[:,1:]]),
        p = torch.vstack([ps, ps]),
        ei = [es[0][:3], es[1][:3], es[0][1:], es[1][1:]],
        ew = [ws[0][:3], ws[1][:3], ws[0][1:], ws[1][1:]]
    )

    check_data([ufld], [dref], label='DynData unfold')

def test_trajmgr_graph(ltg_data):
    metadata = {
        "config" : {
            "data": {
                "path": ltg_data,
                "n_samples": 32,
                "n_steps": 51,
                "double_precision": False},
            "transform_x": [
                {
                "type": "Scaler",
                "mode" : "01"},
                {
                "type": "delay",
                "delay": 2}],
            "transform_u": {
                "type": "Scaler",
                "mode" : "-11"},
            "transform_p": {
                "type": "Scaler",
                "mode" : "std"},
            "transform_ew": {
                "type": "Scaler",
                "mode" : "01"},
    }}

    adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    edge_index, edge_weights = adj_to_edge(adj)

    # --------------------
    # First pass
    tm = TrajectoryManagerGraph(metadata, adj=adj, device="cpu")
    tm.process_all()

    data = np.load(ltg_data, allow_pickle=True)
    ts = data['t']
    xs = data['x']
    us = data['u']
    ps = data['p']

    # ---
    # Forward transform test
    # ---
    # Manually transform data
    # Transform only one node, and then replicate it for all nodes
    trnx = make_transform(metadata['config']['transform_x'])
    trnu = make_transform(metadata['config']['transform_u'])
    trnp = make_transform(metadata['config']['transform_p'])
    trne = make_transform(metadata['config']['transform_ew'])

    ttst = [ts[_i] for _i in tm.test_set_index]

    xtrn = [xs[_i][:,:2] for _i in tm.train_set_index]
    xtst = [xs[_i][:,:2] for _i in tm.test_set_index]
    trnx.fit(xtrn)
    Xtst = trnx.transform(xtst)
    Xtst = [np.concatenate([xt, xt, xt], axis=-1) for xt in Xtst]

    utrn = [us[_i][:,0].reshape(-1,1) for _i in tm.train_set_index]
    utst = [us[_i][:,0].reshape(-1,1) for _i in tm.test_set_index]
    trnu.fit(utrn)
    Utst = trnu.transform(utst)
    Utst = [np.concatenate([ut, ut, ut], axis=-1) for ut in Utst]

    ptrn = [ps[_i][0] for _i in tm.train_set_index]
    ptst = [ps[_i][0] for _i in tm.test_set_index]
    trnp.fit(ptrn)
    Ptst = trnp.transform(ptst)
    Ptst = [np.concatenate([pt, pt, pt], axis=-1) for pt in Ptst]

    itst = [np.tile(edge_index, (len(xs[_i]), 1, 1)).astype(np.int64) for _i in tm.test_set_index]

    etrn = [edge_weights for _ in tm.train_set_index]
    etst = [edge_weights for _ in tm.test_set_index]
    trne.fit(etrn)
    Etst = trne.transform(etst)

    Dtst = [
        DynData(t=tt[2:], x=xt, u=ut[2:], p=pt, ei=ei, ew=ew)
        for tt, xt, ut, pt, ei, ew in zip(ttst, Xtst, Utst, Ptst, itst, Etst)
        ]
    check_data(Dtst, tm.test_set, label='Graph Transform')

    # ---
    # Inverse transform test
    # ---
    # Again manually inverse transform data
    # Inverse transform only one node, and then replicate it for all nodes
    Xrec = trnx.inverse_transform([_d.x[:,:6] for _d in Dtst])
    Xrec = [np.concatenate([xr, xr, xr], axis=-1) for xr in Xrec]
    Urec = trnu.inverse_transform([_d.u[:,0].reshape(-1,1) for _d in Dtst])
    Urec = [np.concatenate([ur, ur, ur], axis=-1) for ur in Urec]
    Prec = trnp.inverse_transform([_d.p[:4] for _d in Dtst])
    Erec = trne.inverse_transform([_d.ew for _d in Dtst])
    Drec = [
        DynData(t=tt, x=xr, u=ur, p=pr, ei=ei, ew=ew)
        for tt, xr, ur, pr, ei, ew in zip(ttst, Xrec, Urec, Prec, itst, Erec)
    ]

    Xref = [xs[_i] for _i in tm.test_set_index]
    Uref = [us[_i][2:] for _i in tm.test_set_index]
    Pref = [ps[_i] for _i in tm.test_set_index]
    Eref = [edge_weights for _ in tm.test_set_index]
    Dref = [
        DynData(t=tr, x=xr, u=ur, p=pr, ei=ei, ew=ew)
        for tr, xr, ur, pr, ei, ew in zip(ttst, Xref, Uref, Pref, itst, Eref)
        ]
    check_data(Drec, Dref, label='Graph Inverse Transform')

    # --------------------
    # Second pass - reinitialize and reload
    old_metadata = copy.deepcopy(tm.metadata)

    new_tm = TrajectoryManagerGraph(old_metadata, adj=adj, device="cpu")
    new_tm.process_all()

    check_data(new_tm.train_set, tm.train_set, label="New Train")
    check_data(new_tm.valid_set, tm.valid_set, label="New Valid")
    check_data(new_tm.test_set,  tm.test_set,  label="New Test")
