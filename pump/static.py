import numpy as np
from qutip import tensor, qeye, sigmaz, sigmam, sigmax, destroy


class DefaultParameters():
    def __init__(self):
        pass


def construct_ham_floquet(params):
    sz = tensor(sigmaz(), qeye(params.c_levels))
    sx = tensor(sigmax(), qeye(params.c_levels))
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels))

    ham0 = (params.fc - params.fp) * a.dag() * a + 0.5 * (params.fa - params.fp) * sz
    ham0 += params.g * (sm * a.dag() + sm.dag() * a)
    ham0 += 0.5 * params.f1 * (sm + sm.dag())
    ham0 *= 2 * np.pi

    return ham0


def construct_ham_floquet_dressed_state(params):
    sz = tensor(sigmaz(), qeye(params.c_levels))
    sx = tensor(sigmax(), qeye(params.c_levels))
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels))

    fap = params.fa - params.fp
    fr = np.sqrt(fap ** 2 + params.f1 ** 2)
    sin_alpha = fap / np.sqrt(fap ** 2 + params.f1 ** 2)
    cos_alpha = params.f1 / np.sqrt(fap ** 2 + params.f1 ** 2)

    ham_coupling = 0
    ham_coupling += 0.5 * params.g * a * (1 + sin_alpha) * sm.dag()
    ham_coupling += 0.5 * params.g * a * (sin_alpha - 1) * sm
    ham_coupling += 0.5 * params.g * a * cos_alpha * sz
    ham_coupling += ham_coupling.dag()

    ham0 = 0.5 * fr * sz + (params.fc - params.fp) * a.dag() * a + ham_coupling
    ham0 *= 2 * np.pi

    return ham0


def construct_ham_floquet_rwa(params):
    sz = tensor(sigmaz(), qeye(params.c_levels))
    sx = tensor(sigmax(), qeye(params.c_levels))
    sm = tensor(sigmam(), qeye(params.c_levels))
    a = tensor(qeye(2), destroy(params.c_levels))

    fap = params.fa - params.fp
    fr = np.sqrt(fap ** 2 + params.f1 ** 2)
    sin_alpha = fap / np.sqrt(fap ** 2 + params.f1 ** 2)
    g_prime = 0.5 * params.g * (sin_alpha + 1)

    ham0 = 0.5 * fr * sz + (params.fc - params.fp) * a.dag() * a + g_prime * (sm * a.dag() + sm.dag() * a)
    ham0 *= 2 * np.pi

    return ham0