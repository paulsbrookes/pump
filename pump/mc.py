from qutip import *
from copy import deepcopy
import numpy as np
import pandas as pd


def construct_ham_floquet(params):
    sz = tensor(sigmaz(), qeye(params['c_levels']))
    sm = tensor(sigmam(), qeye(params['c_levels']))
    a = tensor(qeye(2), destroy(params['c_levels']))

    ham0 = (params['fc'] - params['fp']) * a.dag() * a + 0.5 * (params['fa'] - params['fp']) * sz
    ham0 += params['g'] * (sm * a.dag() + sm.dag() * a)
    ham0 += 0.5 * params['f1'] * (sm + sm.dag())
    ham0 *= 2 * np.pi

    return ham0


def c_ops_gen_jc(params, alpha=0):
    c_ops = []
    sm = tensor(sigmam(), qeye(params['c_levels']))
    a = tensor(qeye(2), destroy(params['c_levels'])) + alpha
    if params['gamma'] > 0.0:
        c_ops.append(np.sqrt(2 * np.pi * params['gamma'] * (1 + params['n_t'])) * sm)
        if params['n_t'] > 0:
            c_ops.append(np.sqrt(2 * np.pi * params['gamma'] * params['n_t']) * sm.dag())
    if params['gamma_phi'] > 0.0:
        c_ops.append(np.sqrt(2 * np.pi * params['gamma_phi']) * sm.dag() * sm)
    if params['kappa'] > 0.0:
        c_ops.append(np.sqrt(2 * np.pi * params['kappa'] * (1 + params['n_c'])) * a)
        if params['n_c'] > 0:
            c_ops.append(np.sqrt(2 * np.pi * params['kappa'] * params['n_c']) * a.dag())
    return c_ops


def ham1_coeff_rw(t, args):
    params = args['params']
    coeff = params['f2'] * np.exp(1j * 2 * np.pi * (params['fd'] - params['fp']) * t)
    return coeff


def ham2_coeff_rw(t, args):
    params = args['params']
    coeff = params['f2'] * np.exp(-1j * 2 * np.pi * (params['fd'] - params['fp']) * t)
    return coeff


def construct_ham(params):
    a = tensor(qeye(2), destroy(params['c_levels']))

    ham0 = construct_ham_floquet(params)

    if params['f2'] is None or params['f2'] == 0:
        return ham0
    else:
        ham1 = 2 * np.pi * a
        ham2 = 2 * np.pi * a.dag()
        H = [ham0, [ham1, ham1_coeff_rw], [ham2, ham2_coeff_rw]]
        return H


def generate_cut_mc(sweep_array, times, params, sweep_param='fd', psi0=None, parallel=False, num_cpus=10, opt=None,
                    process=True):
    task_args = (times,)
    task_kwargs = {'psi0': psi0, 'opt': opt, 'sweep_param': sweep_param, 'process': process}
    params_list = []
    for value in sweep_array:
        params_copy = deepcopy(params)
        params_copy[sweep_param] = value
        params_list.append(params_copy)

    if parallel:
        results = parallel_map(generate_result_mc, params_list, task_args=task_args, task_kwargs=task_kwargs,
                               num_cpus=num_cpus, progress_bar=True)
    else:
        results = [generate_result_mc(p, *task_args, **task_kwargs) for p in params_list]

    results = pd.concat(results)

    return results


def generate_result_mc(params, times, psi0=None, opt=None, e_ops=None, sweep_param='fd', map_func=serial_map,
                       process=True):
    if opt is None:
        opt = Options()
        opt.atol = 1e-6
        opt.rtol = 1e-6
        opt.ntraj = 1

    args = {'params': params}

    if e_ops is None:
        sm = tensor(sigmam(), qeye(params['c_levels']))
        sz = tensor(sigmaz(), qeye(params['c_levels']))
        a = tensor(qeye(2), destroy(params['c_levels']))
        e_ops = {'a': a, 'sm': sm, 'n': a.dag() * a, 'sz': sz}

    c_ops = c_ops_gen_jc(params)

    H = construct_ham(params)
    if psi0 is None:
        rho = steadystate(H[0], c_ops)
        occupations, states = rho.eigenstates()
        psi0 = states[-1]

    trace = mcsolve(H, psi0, times, c_ops, e_ops, args=args, options=opt, map_func=map_func)
    measurement_frequency = params['fd'] - params['fp']
    trace.expect['a'] *= np.exp(1j * 2 * np.pi * measurement_frequency * times)
    trace.expect['sm'] *= np.exp(1j * 2 * np.pi * measurement_frequency * times)

    results_frame = pd.DataFrame(trace.expect, index=times)
    results_frame.index.name = 'time'

    if process:
        measurements = process_trace_mc(results_frame, params)
        measurements[sweep_param] = params[sweep_param]
        measurements.set_index(sweep_param, inplace=True)
        return measurements
    else:
        results_frame[sweep_param] = params[sweep_param]
        results_frame.set_index(sweep_param, inplace=True, append=True)
        results_frame = results_frame.reorder_levels([sweep_param, 'time'])
        return results_frame


def process_cut_mc(cut, start=0.2, stop=1.0, step=1):
    sweep_array = cut.index.levels[0]
    sweep_param = cut.index.names[0]
    measurements = []

    for idx, value in enumerate(sweep_array):
        trace = cut.xs(value, level=sweep_param)
        measurements.append(process_trace_mc(trace, start=start, stop=stop, step=step))

    measurements = pd.concat(measurements)
    measurements.index = sweep_array

    return measurements


def process_trace_mc(trace, start=0.2, stop=1.0, step=1):
    ops = ['a', 'sm', 'n', 'sz']
    measurements = np.zeros(len(ops), dtype=complex)
    n_times = trace.index.shape[0]
    start_idx = int(start * n_times)
    stop_idx = int(stop * n_times)
    trace = trace.iloc[start_idx:stop_idx:step]
    for op_idx, op in enumerate(ops):
        measurements[op_idx] = trace[op].mean()

    measurements = pd.DataFrame([measurements], columns=ops)

    return measurements