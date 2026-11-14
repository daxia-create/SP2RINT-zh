# 中文学习注释：MAPS/Ceviche 第三方核心依赖阅读辅助。未改变上游算法、参数或执行逻辑。
# 模块定位：thirdparty/MAPS_old/thirdparty/ceviche/fdfd.py。重点追踪 Maxwell 离散、线性系统、边界/源项、伴随导数及结构参数化。
# 注意：这里的注释是学习说明，不代表上游项目作者原始注释。

import autograd.numpy as npa
from numpy import indices
import scipy.sparse as sp
import numpy as np
from .constants import *
from .primitives import sp_solve, sp_mult, spsp_mult
from .derivatives import compute_derivative_matrices
from .utils import get_entries_indices

# notataion is similar to that used in: https://www.jpier.org/ac_api/download.php?id=11092006


# [中文注释] 定义 fdfd 类，封装电磁求解/优化相关状态。
class fdfd:
    """Base class for FDFD simulation"""

    # [中文注释] 初始化求解器或优化对象；核对网格、材料、边界和设备参数。
    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        """initialize with a given structure and source
        omega: angular frequency (rad/s)
        dL: grid cell size (m)
        eps_r: array containing relative permittivity
        npml: list of number of PML grid cells in [x, y]
        bloch_{x,y} phase difference across {x,y} boundaries for bloch periodic boundary conditions (default = 0 = periodic)
        """

        self.omega = omega# * 1e6 # if convert to nm, numerical value in A is much smaller
        self.dL = dL# * 1e6
        self.npml = npml

        self._setup_bloch_phases(bloch_phases)

        self.eps_r = eps_r

        self._setup_derivatives()

    """ what happens when you reassign the permittivity of the fdfd object """

    @property
    # [中文注释] 实现 eps_r；结合调用处核对物理量、网格维度、边界条件和单位。
    def eps_r(self):
        """Returns the relative permittivity grid"""
        return self._eps_r

    @eps_r.setter
    # [中文注释] 实现 eps_r；结合调用处核对物理量、网格维度、边界条件和单位。
    def eps_r(self, new_eps):
        """Defines some attributes when eps_r is set."""
        self._save_shape(new_eps)
        self._eps_r = new_eps

    """ classes inherited from fdfd() must implement their own versions of these functions for `fdfd.solve()` to work """

    # [中文注释] 实现 _make_A；结合调用处核对物理量、网格维度、边界条件和单位。
    def _make_A(self, eps_r):
        """This method constucts the entries and indices into the system matrix"""
        raise NotImplementedError("need to make a _make_A() method")

    # [中文注释] 求解离散 Maxwell 线性系统；检查矩阵、右端项、求解器及残差。
    def _solve_fn(self, entries_a, indices_a, source_vec):
        """This method takes the system matrix and source and returns the x, y, and z field components"""
        raise NotImplementedError(
            "need to implement function to solve for field components"
        )

    """ You call this to function to solve for the electromagnetic fields """

    # [中文注释] 求解离散 Maxwell 线性系统；检查矩阵、右端项、求解器及残差。
    def solve(self, source_z):
        """Outward facing function (what gets called by user) that takes a source grid and returns the field components"""

        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec)

        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(
            eps_vec, entries_a, indices_a, source_vec
        )

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        Fz = self._vec_to_grid(Fz_vec)

        return Fx, Fy, Fz

    """ Utility functions for FDFD object """

    # [中文注释] 构造离散导数/Jacobian 算子，为场求解或伴随梯度提供数值基础。
    def _setup_derivatives(self):
        """Makes the sparse derivative matrices and does some processing for ease of use"""

        # Creates all of the operators needed for later
        derivs = compute_derivative_matrices(
            self.omega,
            self.shape,
            self.npml,
            self.dL,
            bloch_x=self.bloch_x,
            bloch_y=self.bloch_y,
        )

        # stores the raw sparse matrices
        self.Dxf, self.Dxb, self.Dyf, self.Dyb, self.Dzf, self.Dzb = derivs        

        # store the entries and elements
        self.entries_Dxf, self.indices_Dxf = get_entries_indices(self.Dxf)
        self.entries_Dxb, self.indices_Dxb = get_entries_indices(self.Dxb)
        self.entries_Dyf, self.indices_Dyf = get_entries_indices(self.Dyf)
        self.entries_Dyb, self.indices_Dyb = get_entries_indices(self.Dyb)
        if self.Dzb is not None:
            self.entries_Dzf, self.indices_Dzf = get_entries_indices(self.Dzf)
            self.entries_Dzb, self.indices_Dzb = get_entries_indices(self.Dzb)
        else:
            self.entries_Dzf = self.indices_Dzf = None
            self.entries_Dzb = self.indices_Dzb = None

        # stores some convenience functions for multiplying derivative matrices by a vector `vec`
        self.sp_mult_Dxf = lambda vec: sp_mult(self.entries_Dxf, self.indices_Dxf, vec)
        self.sp_mult_Dxb = lambda vec: sp_mult(self.entries_Dxb, self.indices_Dxb, vec)
        self.sp_mult_Dyf = lambda vec: sp_mult(self.entries_Dyf, self.indices_Dyf, vec)
        self.sp_mult_Dyb = lambda vec: sp_mult(self.entries_Dyb, self.indices_Dyb, vec)
        if self.Dzf is not None:
            self.sp_mult_Dzf = lambda vec: sp_mult(
                self.entries_Dzf, self.indices_Dzf, vec
            )
            self.sp_mult_Dzb = lambda vec: sp_mult(
                self.entries_Dzb, self.indices_Dzb, vec
            )

    # [中文注释] 实现 _setup_bloch_phases；结合调用处核对物理量、网格维度、边界条件和单位。
    def _setup_bloch_phases(self, bloch_phases):
        """Saves the x y and z bloch phases based on list of them 'bloch_phases'"""

        self.bloch_x = 0.0
        self.bloch_y = 0.0
        self.bloch_z = 0.0
        if bloch_phases is not None:
            self.bloch_x = bloch_phases[0]
            if len(bloch_phases) > 1:
                self.bloch_y = bloch_phases[1]
            if len(bloch_phases) > 2:
                self.bloch_z = bloch_phases[2]

    # [中文注释] 实现 _vec_to_grid；结合调用处核对物理量、网格维度、边界条件和单位。
    def _vec_to_grid(self, vec):
        """converts a vector quantity into an array of the shape of the FDFD simulation"""
        return npa.reshape(vec, self.shape)

    # [中文注释] 实现 _grid_to_vec；结合调用处核对物理量、网格维度、边界条件和单位。
    def _grid_to_vec(self, grid):
        """converts a grid of the shape of the FDFD simulation to a flat vector"""
        return grid.flatten()

    # [中文注释] 实现 _save_shape；结合调用处核对物理量、网格维度、边界条件和单位。
    def _save_shape(self, grid):
        """Sores the shape and size of `grid` array to the FDFD object"""
        self.shape = grid.shape
        self.Nx, self.Ny = self.shape[:2]
        if len(self.shape) > 2:
            self.Nz = self.shape[-1]
        else:
            self.Nz = 1
        self.N = np.prod(self.shape)

    @staticmethod
    # [中文注释] 实现 _default_val；结合调用处核对物理量、网格维度、边界条件和单位。
    def _default_val(val, default_val=None):
        # not used yet
        return val if val is not None else default_val

    """ Field conversion functions for 2D.  Function names are self explanatory """

    # [中文注释] 实现 _Ex_Ey_to_Hz；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Ex_Ey_to_Hz(self, Ex_vec, Ey_vec):
        return (
            1
            / 1j
            / self.omega
            / MU_0
            * (self.sp_mult_Dxb(Ey_vec) - self.sp_mult_Dyb(Ex_vec))
        )

    # [中文注释] 实现 _Ez_to_Hx；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Ez_to_Hx(self, Ez_vec):
        return -1 / 1j / self.omega / MU_0 * self.sp_mult_Dyb(Ez_vec)

    # [中文注释] 实现 _Ez_to_Hy；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Ez_to_Hy(self, Ez_vec):
        return 1 / 1j / self.omega / MU_0 * self.sp_mult_Dxb(Ez_vec)

    # [中文注释] 实现 _Ez_to_Hx_Hy；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Ez_to_Hx_Hy(self, Ez_vec):
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    # addition of 1e-5 is for numerical stability when tracking gradients of eps_xx, and eps_yy -> 0
    # [中文注释] 实现 _Hz_to_Ex；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Hz_to_Ex(self, Hz_vec, eps_vec_xx):
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_xx + 1e-5)
            * self.sp_mult_Dyf(Hz_vec)
        )

    # [中文注释] 实现 _Hz_to_Ey；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Hz_to_Ey(self, Hz_vec, eps_vec_yy):
        return (
            -1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_yy + 1e-5)
            * self.sp_mult_Dxf(Hz_vec)
        )

    # [中文注释] 实现 _Hx_Hy_to_Ez；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Hx_Hy_to_Ez(self, Hx_vec, Hy_vec, eps_vec_zz):
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_zz + 1e-5)
            * (self.sp_mult_Dxf(Hy_vec) - self.sp_mult_Dyf(Hx_vec))
        )

    # [中文注释] 实现 _Hz_to_Ex_Ey；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Hz_to_Ex_Ey(self, Hz_vec, eps_vec_xx, eps_vec_yy):
        Ex_vec = self._Hz_to_Ex(Hz_vec, eps_vec_xx)
        Ey_vec = self._Hz_to_Ey(Hz_vec, eps_vec_yy)
        return Ex_vec, Ey_vec


""" These are the fdfd classes that you'll actually want to use """


# [中文注释] 定义 fdfd_ez 类，封装电磁求解/优化相关状态。
class fdfd_ez(fdfd):
    """FDFD class for linear Ez polarization"""

    # [中文注释] 初始化求解器或优化对象；核对网格、材料、边界和设备参数。
    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

    # [中文注释] 实现 _make_A；结合调用处核对物理量、网格维度、边界条件和单位。
    def _make_A(self, eps_vec):
        C = -1 / MU_0 * self.Dxf.dot(self.Dxb) - 1 / MU_0 * self.Dyf.dot(self.Dyb)
        entries_c, indices_c = get_entries_indices(C)

        # indices into the diagonal of a sparse matrix
        entries_diag = -EPSILON_0 * self.omega**2 * eps_vec
        indices_diag = npa.vstack((npa.arange(self.N), npa.arange(self.N)))

        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))

        return entries_a, indices_a

    # [中文注释] 求解离散 Maxwell 线性系统；检查矩阵、右端项、求解器及残差。
    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec):
        b_vec = 1j * self.omega * Jz_vec
        Ez_vec = sp_solve(entries_a, indices_a, b_vec)
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec


# [中文注释] 定义 fdfd_hz 类，封装电磁求解/优化相关状态。
class fdfd_hz(fdfd):
    """FDFD class for linear Ez polarization"""

    # [中文注释] 初始化求解器或优化对象；核对网格、材料、边界和设备参数。
    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

    # [中文注释] 实现 _grid_average_2d；结合调用处核对物理量、网格维度、边界条件和单位。
    def _grid_average_2d(self, eps_vec):
        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=1))
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        return eps_vec_xx, eps_vec_yy

    # [中文注释] 实现 _make_A；结合调用处核对物理量、网格维度、边界条件和单位。
    def _make_A(self, eps_vec):
        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)
        eps_vec_xx_inv = 1 / (eps_vec_xx + 1e-5)  # the 1e-5 is for numerical stability
        eps_vec_yy_inv = 1 / (
            eps_vec_yy + 1e-5
        )  # autograd throws 'divide by zero' errors.

        indices_diag = npa.vstack((npa.arange(self.N), npa.arange(self.N)))

        entries_DxEpsy, indices_DxEpsy = spsp_mult(
            self.entries_Dxb, self.indices_Dxb, eps_vec_yy_inv, indices_diag, self.N
        )
        entires_DxEpsyDx, indices_DxEpsyDx = spsp_mult(
            entries_DxEpsy, indices_DxEpsy, self.entries_Dxf, self.indices_Dxf, self.N
        )

        entries_DyEpsx, indices_DyEpsx = spsp_mult(
            self.entries_Dyb, self.indices_Dyb, eps_vec_xx_inv, indices_diag, self.N
        )
        entires_DyEpsxDy, indices_DyEpsxDy = spsp_mult(
            entries_DyEpsx, indices_DyEpsx, self.entries_Dyf, self.indices_Dyf, self.N
        )

        entries_d = 1 / EPSILON_0 * npa.hstack((entires_DxEpsyDx, entires_DyEpsxDy))
        indices_d = npa.hstack((indices_DxEpsyDx, indices_DyEpsxDy))

        entries_diag = MU_0 * self.omega**2 * npa.ones(self.N)

        entries_a = npa.hstack((entries_d, entries_diag))
        indices_a = npa.hstack((indices_d, indices_diag))

        return entries_a, indices_a

    # [中文注释] 求解离散 Maxwell 线性系统；检查矩阵、右端项、求解器及残差。
    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec):
        b_vec = 1j * self.omega * Mz_vec  # needed so fields are SI units
        Hz_vec = sp_solve(entries_a, indices_a, b_vec)
        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)

        # strip out the x and y components of E and find the Hz component
        Ex_vec, Ey_vec = self._Hz_to_Ex_Ey(Hz_vec, eps_vec_xx, eps_vec_yy)

        return Ex_vec, Ey_vec, Hz_vec


# [中文注释] 定义 fdfd_mf_ez 类，封装电磁求解/优化相关状态。
class fdfd_mf_ez(fdfd):
    """FDFD class for multifrequency linear Ez polarization. New variables:
    omega_mod: angular frequency of modulation (rad/s)
    delta: array of shape (Nfreq, Nx, Ny) containing pointwise modulation depth for each modulation harmonic (1,...,Nfreq)
    phi: array of same shape as delta containing pointwise modulation phase for each modulation harmonic
    Nsb: number of numerical sidebands to consider when solving for fields.
    This is not the same as the number of modulation frequencies Nfreq. For physically meaningful results, Nsb >= Nfreq.
    """

    # [中文注释] 初始化求解器或优化对象；核对网格、材料、边界和设备参数。
    def __init__(
        self, omega, dL, eps_r, omega_mod, delta, phi, Nsb, npml, bloch_phases=None
    ):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)
        self.omega_mod = omega_mod
        self.delta = delta
        self.phi = phi
        self.Nsb = Nsb

    # [中文注释] 求解离散 Maxwell 线性系统；检查矩阵、右端项、求解器及残差。
    def solve(self, source_z):
        """Outward facing function (what gets called by user) that takes a source grid and returns the field components"""
        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)
        Nfreq = npa.shape(self.delta)[0]
        delta_matrix = self.delta.reshape([Nfreq, npa.prod(self.shape)])
        phi_matrix = self.phi.reshape([Nfreq, npa.prod(self.shape)])
        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec, delta_matrix, phi_matrix)

        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(
            eps_vec, entries_a, indices_a, source_vec
        )

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        Fz = self._vec_to_grid(Fz_vec)

        return Fx, Fy, Fz

    # [中文注释] 实现 _make_A；结合调用处核对物理量、网格维度、边界条件和单位。
    def _make_A(self, eps_vec, delta_matrix, phi_matrix):
        """Builds the multi-frequency electromagnetic operator A in Ax = b"""
        M = 2 * self.Nsb + 1
        N = self.Nx * self.Ny
        W = self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod

        C = sp.kron(
            sp.eye(M),
            -1 / MU_0 * self.Dxf.dot(self.Dxb) - 1 / MU_0 * self.Dyf.dot(self.Dyb),
        )
        entries_c, indices_c = get_entries_indices(C)

        # diagonal entries representing static refractive index
        # this part is just a block diagonal version of the single frequency fdfd_ez
        entries_diag = -EPSILON_0 * npa.kron(W**2, eps_vec)
        indices_diag = npa.vstack((npa.arange(M * N), npa.arange(M * N)))

        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))

        # off-diagonal entries representing dynamic modulation
        # this part couples different frequencies due to modulation
        # for a derivation of these entries, see Y. Shi, W. Shin, and S. Fan. Optica 3(11), 2016.
        Nfreq = npa.shape(delta_matrix)[0]
        for k in npa.arange(Nfreq):
            # super-diagonal entries (note the +1j phase)
            mod_p = (
                -0.5 * EPSILON_0 * delta_matrix[k, :] * npa.exp(1j * phi_matrix[k, :])
            )
            entries_p = npa.kron(W[: -k - 1] ** 2, mod_p)
            indices_p = npa.vstack(
                (npa.arange((M - k - 1) * N), npa.arange((k + 1) * N, M * N))
            )
            entries_a = npa.hstack((entries_p, entries_a))
            indices_a = npa.hstack((indices_p, indices_a))
            # sub-diagonal entries (note the -1j phase)
            mod_m = (
                -0.5 * EPSILON_0 * delta_matrix[k, :] * npa.exp(-1j * phi_matrix[k, :])
            )
            entries_m = npa.kron(W[k + 1 :] ** 2, mod_m)
            indices_m = npa.vstack(
                (npa.arange((k + 1) * N, M * N), npa.arange((M - k - 1) * N))
            )
            entries_a = npa.hstack((entries_m, entries_a))
            indices_a = npa.hstack((indices_m, indices_a))

        return entries_a, indices_a

    # [中文注释] 求解离散 Maxwell 线性系统；检查矩阵、右端项、求解器及残差。
    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec):
        """Multi-frequency version of _solve_fn() defined in fdfd_ez"""
        M = 2 * self.Nsb + 1
        N = self.Nx * self.Ny
        W = self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod
        P = sp.kron(sp.spdiags(W, [0], M, M), sp.eye(N))
        entries_p, indices_p = get_entries_indices(P)
        b_vec = 1j * sp_mult(entries_p, indices_p, Jz_vec)
        Ez_vec = sp_solve(entries_a, indices_a, b_vec)
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec

    # [中文注释] 实现 _Ez_to_Hx；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Ez_to_Hx(self, Ez_vec):
        """Multi-frequency version of _Ez_to_Hx() defined in fdfd"""
        M = 2 * self.Nsb + 1
        Winv = 1 / (self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod)
        Dyb_mf = sp.kron(sp.spdiags(Winv, [0], M, M), self.Dyb)
        entries_Dyb_mf, indices_Dyb_mf = get_entries_indices(Dyb_mf)
        return -1 / 1j / MU_0 * sp_mult(entries_Dyb_mf, indices_Dyb_mf, Ez_vec)

    # [中文注释] 实现 _Ez_to_Hy；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Ez_to_Hy(self, Ez_vec):
        """Multi-frequency version of _Ez_to_Hy() defined in fdfd"""
        M = 2 * self.Nsb + 1
        Winv = 1 / (self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod)
        Dxb_mf = sp.kron(sp.spdiags(Winv, [0], M, M), self.Dxb)
        entries_Dxb_mf, indices_Dxb_mf = get_entries_indices(Dxb_mf)
        return 1 / 1j / MU_0 * sp_mult(entries_Dxb_mf, indices_Dxb_mf, Ez_vec)

    # [中文注释] 实现 _Ez_to_Hx_Hy；结合调用处核对物理量、网格维度、边界条件和单位。
    def _Ez_to_Hx_Hy(self, Ez_vec):
        """Multi-frequency version of _Ez_to_Hx_Hy() defined in fdfd"""
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    # [中文注释] 实现 _vec_to_grid；结合调用处核对物理量、网格维度、边界条件和单位。
    def _vec_to_grid(self, vec):
        """Multi-frequency version of _vec_to_grid() defined in fdfd"""
        # grid shape has Nx*Ny cells per frequency sideband
        grid_shape = (2 * self.Nsb + 1, self.Nx, self.Ny)
        return npa.reshape(vec, grid_shape)


# [中文注释] 定义 fdfd_ez_3d 类，封装电磁求解/优化相关状态。
class fdfd_ez_3d(fdfd):
    """3D FDFD class (work in progress)"""

    # [中文注释] 初始化求解器或优化对象；核对网格、材料、边界和设备参数。
    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

    # [中文注释] 实现 _grid_average_3d；结合调用处核对物理量、网格维度、边界条件和单位。
    def _grid_average_3d(self, eps_vec, dir: str):
        ## Input Parameters
        # center_array: 3D array of values defined at cell centers
        # w: 'x' or 'y' or 'z' direction in which average is taken

        ## Out Parameter
        # avg_array: 2D array of averaged values
        eps = eps_vec.reshape(self.shape)
        if dir == "x":
            shift = 1
        if dir == "y":
            shift = 2
        if dir == "z":
            shift = 3

        center_shifted = np.roll(
            eps, shift
        )  # doe sthis generalize easily into 3 Dimensions, CHECK!
        avg_array = (center_shifted + eps) / 2
        return avg_array.flatten()

    # [中文注释] 实现 _make_A；结合调用处核对物理量、网格维度、边界条件和单位。
    def _make_A(self, eps_vec):
        # curlE and curlH
        Ce = sp.bmat(
            [
                [None, -self.Dzf, self.Dyf],
                [self.Dzf, None, -self.Dxf],
                [-self.Dyf, self.Dxf, None],
            ]
        )
        Ch = sp.bmat(
            [
                [None, -self.Dzb, self.Dyb],
                [self.Dzb, None, -self.Dxb],
                [-self.Dyb, self.Dxb, None],
            ]
        )
        C = 1 / MU_0 * Ch @ Ce
        entries_c, indices_c = get_entries_indices(C)
        entries_diag = -EPSILON_0 * self.omega**2 * np.concatenate([eps_vec] * 3)
        indices_diag = np.vstack([np.arange(3 * self.N), np.arange(3 * self.N)])

        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))

        return entries_a, indices_a

    # [中文注释] 求解离散 Maxwell 线性系统；检查矩阵、右端项、求解器及残差。
    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec):
        # convert the Mz current into Jx, Jy
        b_vec = 1j * self.omega * Jz_vec
        Ez_vec = sp_solve(entries_a, indices_a, b_vec)
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec
