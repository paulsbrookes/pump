import numpy as np
import pandas as pd
from copy import deepcopy
from .static import construct_ham_floquet
from qutip import tensor, fock_dm, sigmaz, sigmax, sigmam, destroy, qeye, parallel_map, Options, mesolve, steadystate


def c_ops_gen_jc(params, alpha=0):
    c_ops = []
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels)) + alpha
    if params.gamma > 0.0:
        c_ops.append(np.sqrt(2 * np.pi * params.gamma * (1 + params.n_t)) * sm)
        if params.n_t > 0:
            c_ops.append(np.sqrt(2 * np.pi * params.gamma * params.n_t) * sm.dag())
    if params.gamma_phi > 0.0:
        c_ops.append(np.sqrt(2 * np.pi * params.gamma_phi) * sm.dag() * sm)
    if params.kappa > 0.0:
        c_ops.append(np.sqrt(2 * np.pi * params.kappa * (1 + params.n_c)) * a)
        if params.n_c > 0:
            c_ops.append(np.sqrt(2 * np.pi * params.kappa * params.n_c) * a.dag())
    return c_ops


def ham1_coeff_rw(t, args):
    params = args['params']
    coeff = params.f2 * np.exp(1j * 2 * np.pi * (params.fd - params.fp) * t)
    return coeff


def ham2_coeff_rw(t, args):
    params = args['params']
    coeff = params.f2 * np.exp(-1j * 2 * np.pi * (params.fd - params.fp) * t)
    return coeff


def construct_ham(params):
    a = tensor(qeye(2), destroy(params.c_levels))

    ham0 = construct_ham_floquet(params)

    if params.f2 is None or params.f2 == 0:
        return ham0
    else:
        ham1 = 2 * np.pi * a
        ham2 = 2 * np.pi * a.dag()
        H = [ham0, [ham1, ham1_coeff_rw], [ham2, ham2_coeff_rw]]
        return H


def generate_cut_states(fd_array, times, params, psi0=None, parallel=False, num_cpus=10, opt=None):
    task_args = (times,)
    task_kwargs = {'psi0': psi0, 'opt': opt}
    params_list = []
    for fd in fd_array:
        params_copy = deepcopy(params)
        params_copy.fd = fd
        params_list.append(params_copy)

    if parallel:
        results = parallel_map(generate_result_states, params_list, task_args=task_args, task_kwargs=task_kwargs,
                               num_cpus=num_cpus, progress_bar=True)
    else:
        results = [generate_result(p, *task_args, **task_kwargs) for p in params_list]

    results = pd.concat(results)

    return results


def transitions_calc(params):
    ham = construct_ham_floquet(params)
    energies, states = ham.eigenstates()
    energies /= (2 * np.pi)
    transitions_0 = params.fp + energies[2] - energies[0]
    transitions_1 = params.fp + energies[3] - energies[1]
    return transitions_0, transitions_1


def generate_result_states(params, times, psi0=None, opt=None):
    if opt is None:
        opt = Options()
        opt.atol = 1e-6
        opt.rtol = 1e-6

    args = {'params': params}

    c_ops = c_ops_gen_jc(params)
    e_ops = []

    H = construct_ham(params)
    if psi0 is None:
        psi0 = steadystate(H[0], c_ops)

    opt.store_states = True
    trace = mesolve(H, psi0, times, c_ops, e_ops, args=args, options=opt)

    size = 2 * params.c_levels
    state_array = np.zeros([len(trace.states), size, size], dtype=complex)
    for i, state in enumerate(trace.states):
        state_array[i] = state.full()

    frame_content = [[trace.times, state_array, params]]
    columns = ['times', 'states', 'params']
    results_frame = pd.DataFrame(frame_content, columns=columns)
    results_frame['fd'] = params.fd
    results_frame.set_index('fd', inplace=True)

    return results_frame


def expect_array(e_op, states):
    expectations = np.trace(e_op @ states, axis1=-2, axis2=-1)
    return expectations


def gen_projectors(params):
    ham = construct_ham_floquet(params)
    energies, states = ham.eigenstates()
    P_0, P_1 = 0, 0
    for i, s in enumerate(states):
        if i % 2 == 0:
            P_0 += s * s.dag()
        else:
            P_1 += s * s.dag()
    return P_0, P_1


def expectations_breakdown(times, states, e_op, params):
    P_0, P_1 = gen_projectors(params)
    e_op_0 = P_0 * e_op * P_0
    e_op_1 = P_1 * e_op * P_1
    e_op_cross = P_0 * e_op * P_1 + P_1 * e_op * P_0
    expectations_0 = expect_array(e_op_0, states)
    expectations_1 = expect_array(e_op_1, states)
    expectations_cross = expect_array(e_op_cross, states)
    expectations_total = expect_array(e_op, states)
    expectations = np.array([expectations_0, expectations_1, expectations_cross, expectations_total]).T
    expectations = pd.DataFrame(expectations, index=times, columns=['0', '1', 'cross', 'total'])
    expectations.index.name = 'time'
    return expectations


def process_trace_states(trace, start=0.9, stop=1.0):
    a = tensor(qeye(2), destroy(trace.params.c_levels))
    n_times = trace.times.shape[0]
    start_idx = int(start * n_times)
    stop_idx = int(stop * n_times)
    expectations = expectations_breakdown(trace.times[start_idx:stop_idx], trace.states[start_idx:stop_idx], a, trace.params)
    measurement_frequency = trace.params.fd - trace.params.fp
    phases = np.exp(1j * 2 * np.pi * measurement_frequency * np.array(expectations.index))
    expectations *= phases[:, np.newaxis]
    expectations = expectations.mean()
    return expectations


def generate_result(params, times, psi0=None, opt=None, e_ops=None):
    if opt is None:
        opt = Options()
        opt.atol = 1e-6
        opt.rtol = 1e-6

    args = {'params': params}

    if e_ops is None:
        e_ops = e_ops_gen(params)

    c_ops = c_ops_gen_jc(params)

    H = construct_ham(params)
    if psi0 is None:
        psi0 = steadystate(H[0], c_ops)

    trace = mesolve(H, psi0, times, c_ops, e_ops, args=args, options=opt)

    results_frame = pd.DataFrame(trace.expect, index=times)
    results_frame.index.name = 'time'
    results_frame['fd'] = params.fd
    results_frame.set_index('fd', inplace=True, append=True)
    results_frame = results_frame.reorder_levels(['fd', 'time'])

    return results_frame


def e_ops_gen(params):

    projectors = gen_projectors(params)

    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels))

    base_e_ops = {'a': a, 'sm': sm, 'n': a.dag() * a}
    e_ops = base_e_ops.copy()

    for key, item in base_e_ops.items():
        for i in range(2):
            for j in range(2):
                e_ops[key + '_' + str(i) + str(j)] = projectors[i] * item * projectors[j]

    for n in range(2):
        e_ops['p_q' + str(n)] = tensor(fock_dm(2, n), qeye(params.c_levels))

    for n in range(params.c_levels):
        e_ops['p_c' + str(n)] = tensor(qeye(2), fock_dm(params.c_levels, n))

    e_ops['P_0'] = projectors[0]
    e_ops['P_1'] = projectors[1]
    e_ops['P_01'] = projectors[0] * projectors[1]

    return e_ops


def generate_cut(fd_array, times, params, psi0=None, parallel=False, num_cpus=10, opt=None):
    task_args = (times,)
    task_kwargs = {'psi0': psi0, 'opt': opt}
    params_list = []
    for fd in fd_array:
        params_copy = deepcopy(params)
        params_copy.fd = fd
        params_list.append(params_copy)

    if parallel:
        results = parallel_map(generate_result, params_list, task_args=task_args, task_kwargs=task_kwargs,
                               num_cpus=num_cpus, progress_bar=True)
    else:
        results = [generate_result(p, *task_args, **task_kwargs) for p in params_list]

    results = pd.concat(results)

    return results


def process_trace(trace, modulation=0):
    value = (trace * np.exp(1j * 2 * np.pi * modulation * trace.index.values)).mean()
    return value


def process_cut(cut, start=0.9, stop=1.0, step=1):
    ops = ['a', 'a_00', 'a_11', 'a_01', 'a_10']
    fd_array = cut.index.levels[0]
    measurements = np.zeros([fd_array.shape[0], len(ops)], dtype=complex)

    for fd_idx, fd in enumerate(fd_array):
        trace = cut.xs(fd, level='fd')
        n_times = trace.index.shape[0]
        start_idx = int(start * n_times)
        stop_idx = int(stop * n_times)
        trace = trace.iloc[start_idx:stop_idx:step]
        modulation = fd - params.fp
        for op_idx, op in enumerate(ops):
            measurements[fd_idx, op_idx] = process_trace(trace[op], modulation=modulation)

    measurements = pd.DataFrame(measurements, columns=ops, index=fd_array)

    return measurements
