# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet_utils.py

import torch
from math import pi as PI
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import torch
from math import pi as PI
import sympy as sym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def swish(x):
    return x * torch.sigmoid(x)


def gather_matrix(a, b):
    index = a.long()
    b_expanded = b[index.unsqueeze(-1), :]
    c = b_expanded.squeeze(-2).sum(-2)
    return c


def quaternion_split(x):
    num_edge, n, _ = x.shape
    x = x.permute(0, 2, 1).reshape(num_edge, n * 4)
    r, i, j, k = x.unsqueeze(-2).split(x.shape[-1] // 4, dim=-1)
    return r, i, j, k


def qpu_linear(input, weight, bias):
    """
    input: (*, C_in * 4) -> (*, C_out * 4)
    """
    # print(weight.shape) # [64, 25]
    in_channels = input.shape[-1]  # 100
    in_channels = in_channels // 4  # 消去quaternion 4维数据的影响, 25
    out_channels = weight.shape[0]  # 64
    # print(input.unsqueeze(-2).shape) # [1280, 1, 100]
    r, i, j, k = input.unsqueeze(-2).split(in_channels, dim=-1)
    # print(r.shape) # [1280, 1, 25]
    r, i, j, k = quaternion_power_bias(r, i, j, k, weight, bias)
    # [1280, 64, 25]
    r, i, j, k = QuaternionRemoveZeros.apply(r, i, j, k)
    r, i, j, k = hamilton_chained_product(r, i, j, k, -1)
    # We can also use the custom autograd function which significantly decrease GPU memory usage (when QPU layers become deep), but is slower.
    # r, i, j, k = QuaternionChainedProdFunction.apply(r, i, j, k, -1)

    return torch.cat([r, i, j, k], dim=-1)


class QuaternionRemoveZeros(torch.autograd.Function):
    """Replace [0, 0, 0, 0] with [1, 0, 0, 0]
    """

    @staticmethod
    def forward(ctx, r, i, j, k):
        norm = r ** 2 + i ** 2 + j ** 2 + k ** 2
        index = norm == 0
        ctx.save_for_backward(index)
        r[index] = 1
        return r, i, j, k

    @staticmethod
    def backward(ctx, gr, gi, gj, gk):
        index, = ctx.saved_tensors
        gr[index] = 0
        gi[index] = 0
        gj[index] = 0
        gk[index] = 0
        return gr, gi, gj, gk


def quaternion_normalize(input, dim):
    """ Normalize quaternion
    """
    in_channels = input.shape[dim] // 4
    r, i, j, k = input.split(in_channels, dim)
    norm = torch.sqrt(r ** 2 + i ** 2 + j ** 2 + k ** 2 + 1e-12)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm
    return torch.cat([r, i, j, k], dim=dim)


def quaternion_power_bias(r, i, j, k, weight, bias):
    """
    r, i, j, k: (*, 1, C_in)
    weight: (C_out, C_in)
    bias: (C_out)
    return: [cos(w * (acos(r) + bias)), sin(w * (acos(r) + bias)) v / |v|]
    """
    # Compute new theta
    norm_v = torch.sqrt(i ** 2 + j ** 2 + k ** 2 + 1e-12)
    # print(norm_v.shape) # [1280, 1, 25]
    theta = torch.acos(torch.clamp(r, min=-1 + 1e-6, max=1 - 1e-6))
    # print(theta.shape) # [1280, 1, 25]
    # print(bias.unsqueeze(-1).shape) # [64, 1]
    if bias is not None:
        theta = theta + bias.unsqueeze(-1)
        # print(theta.shape) # [1280, 64, 25]
    theta = weight * theta  # 对应位置相乘

    mul = torch.sin(theta) / norm_v
    # print(mul.shape) # [1280, 64, 25]
    # print(i.shape) # [1280, 1, 25]
    r = torch.cos(theta)
    i = i * mul
    j = j * mul
    k = k * mul
    # print(i.shape) # [1280, 64, 25]
    return r, i, j, k


def quaternion_power(r, i, j, k, w):
    """
    r, i, j, k: (..., C_in, ...)
    w: (..., C_in, ...)
    return: [cos(w * acos(r)), sin(w * acos(r)) v / |v|]
    """
    # Compute new theta
    norm_v = torch.sqrt(i ** 2 + j ** 2 + k ** 2 + 1e-12)
    theta = w * torch.acos(torch.clamp(r, min=-1 + 1e-6, max=1 - 1e-6))
    # Compute new quaternion
    r = torch.cos(theta)
    mul = torch.sin(theta) / norm_v
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k


def hamilton_product_chunk(r1, i1, j1, k1, r2, i2, j2, k2):
    """
    Hamilton product
    a1 a2 - b1 b2 - c1 c2 - d1 d2
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k
    """
    # print(r1.shape) # [1280, 64, 12]
    r_out, i_out, j_out, k_out = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2, \
                                 r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2, \
                                 r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2, \
                                 r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
    return r_out, i_out, j_out, k_out


def hamilton_chained_product(r_input, i_input, j_input, k_input, dim, last=None):
    """
    Chained quaternion product along a dimension (recursive)
    Hamilton product:
    a1 a2 - b1 b2 - c1 c2 - d1 d2
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k
    """
    channel = r_input.shape[dim]  # 25
    # print(r_input.shape) # [1280, 64, 25]
    if channel == 1:
        return r_input.squeeze(dim), i_input.squeeze(dim), j_input.squeeze(dim), k_input.squeeze(dim)
    else:
        # Split into pair(0) and odd(1)
        r_out, i_out, j_out, k_out = r_input.unfold(dim, 2, 2), i_input.unfold(dim, 2, 2), j_input.unfold(dim, 2,
                                                                                                          2), k_input.unfold(
            dim, 2, 2)
        # print(r_out.shape) # [1280, 64, 12, 2]
        r_pair, r_odd = r_out.select(-1, 0), r_out.select(-1, 1)
        i_pair, i_odd = i_out.select(-1, 0), i_out.select(-1, 1)
        j_pair, j_odd = j_out.select(-1, 0), j_out.select(-1, 1)
        k_pair, k_odd = k_out.select(-1, 0), k_out.select(-1, 1)
        # print(r_pair.shape) # [1280, 64, 12]
        # pair * odd
        r_out, i_out, j_out, k_out = hamilton_product_chunk(r_pair, i_pair, j_pair, k_pair, r_odd, i_odd, j_odd, k_odd)
        # print(r_out.shape) # [1280, 64, 12]
        # Multiply last
        if channel % 2 == 1:
            last = (r_input.select(dim, -1), i_input.select(dim, -1), j_input.select(dim, -1), k_input.select(dim, -1))
            # print(last[0].shape) # [1280, 64]
        if r_out.shape[dim] % 2 == 1 and last is not None:
            r_out = torch.cat([r_out, last[0].unsqueeze(dim)], dim=dim)
            i_out = torch.cat([i_out, last[1].unsqueeze(dim)], dim=dim)
            j_out = torch.cat([j_out, last[2].unsqueeze(dim)], dim=dim)
            k_out = torch.cat([k_out, last[3].unsqueeze(dim)], dim=dim)
            last = None
        # Recursion
        r_out, i_out, j_out, k_out = hamilton_chained_product(r_out, i_out, j_out, k_out, dim, last)
        return r_out, i_out, j_out, k_out


def quaternion_product(quaternion):
    r, i, j, k = quaternion_split(quaternion)  # [8000, 1, 30]
    r, i, j, k = QuaternionRemoveZeros.apply(r, i, j, k)
    r, i, j, k = hamilton_chained_product(r, i, j, k, -1)  # [8000, 1]
    return r, i, j, k


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    x = sym.symbols('x')

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(k, m):
    return ((2 * k + 1) * np.math.factorial(k - abs(m)) /
            (4 * np.pi * np.math.factorial(k + abs(m)))) ** 0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] -
                                        (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))

    return P_l_m


def real_sph_harm(l, zero_m_only=False, spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        x = sym.symbols('x')
        y = sym.symbols('y')
        S_m = [x * 0]
        C_m = [1 + 0 * x]
        # S_m = [0]
        # C_m = [1]
        for i in range(1, l):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(
                    theta) * sym.cos(phi)).subs(y, sym.sin(theta) * sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(
                    theta) * sym.cos(phi)).subs(y, sym.sin(theta) * sym.sin(phi))

    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class dist_emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(dist_emb, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        self.freq.data = torch.arange(1, self.freq.numel() + 1).float().mul_(PI)

    def forward(self, dist):
        # print("dist1 shape", dist.shape) # [_]
        dist = dist.unsqueeze(-1) / self.cutoff
        # print("dist2 shape", dist.shape) # [_ , 1]
        # print("self.freq * dist shape", (self.freq * dist).shape) # [_ , 6]
        return self.envelope(dist) * (self.freq * dist).sin()  # [_ , 1] * [_ , 6] = [_ , 6]


class angle_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)
        # print("angle_sph", self.sph_funcs)
        # print(len(self.sph_funcs)) # 3
        # print(len(self.bessel_funcs)) # 18 = 3 * 6
        # print("##########################################")

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        # print("dist.shape", dist.shape) # [9064]
        # print("angle.shape", angle.shape) # [136470]
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # print("rbf.shape", rbf.shape) # [9064 , 18]
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        # print("cbf.shape", cbf.shape) # [136470 , 3]
        n, k = self.num_spherical, self.num_radial
        # print("idx_kj.shape", idx_kj.shape) # [136470]
        # print("rbf[idx_kj].shape", rbf[idx_kj].shape) # [136470 , 18]
        # print("rbf[idx_kj].view(-1, n, k).shape", rbf[idx_kj].view(-1, n, k).shape) # [136470 , 3 , 6]
        # print("cbf.view(-1, n, 1)).shape", cbf.view(-1, n, 1).shape) # [136470 , 3 , 1]
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        # print("angle_emb.shape", out.shape) # [136470 , 18]
        return out


class torsion_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical  #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: torch.zeros_like(x) + torch.zeros_like(y) + sph1(0,
                                                                                                    0))  # torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k + i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)
        # print("len(self.sph_funcs)", len(self.sph_funcs)) # 9

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        # print("torsion_emb.shape", out.shape) # [136470, 54]
        return out