#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/12/14 14:16:35
@modified  :   Lu Zhanglin

code aim to: 2D顶盖驱动空腔流模拟-交错网格离散格式

离散方程求解方法: 双向共轭梯度法
"""


import taichi as ti
from display import Display
from cgsolver import CGSolver
from bicgsolver import BICGSolver

ti.init(arch=ti.cpu, default_fp=ti.f64)


@ti.data_oriented
class SIMPLESolver:
    """2D lid-driven cavity flow-multigrid 顶盖驱动空腔流模拟-交错网格离散格式

    Step 1. 假定速度分布, u0, v0, 依次计算动量离散方程中的系数和常数项;\\
    Step 2. 假定压力场p*;\\
    Step 3. 依次求解两个动量方程, 得u*, v*;\\
    Step 4. 求解压力修正方程, 得p';\\
    Step 5. 据p'改进压力场、速度场;\\
    Step 6. 利用改进后的速度场求解那些通过源项物性等与速度场耦合的Φ变量, 如果Φ不影响流场, 则应在速度场收敛后再求解;\\
    Step 7. 利用改进后的速度场重新计算动量离散方程的系数, 并用改进后的压力长作为下一层次迭代计算的初值\\
    重复上述步骤, 直到收敛

    Args:
        lx (float): x length of cavity
        ly (float): y length of cavity
        nx (float): number of pressure nodes in x axis
        yx (float): number of pressure nodes in y axis

    Attributes:
        rho (float): density of flow, set to 1.0
        mu (float): dynamic viscosity, set to 0.01
        dt (float): discrete time, set to 1e16
        Re (float): Re = \rho ·v·L / mu
    """

    def __init__(self, lx, ly, nx, ny):
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        # 密度
        self.rho = 1.00
        # 摩擦力
        self.mu = 0.01
        # 1/0.01=100 1 m/s时雷诺数=100
        self.Re = self.rho * 1 * self.lx / self.mu
        self.dt = 1e16

        """对于正交网格, 建议 alpha_u + alpha_p = c, c取1或1.1
        alpha_u尽可能大, 一般取0.7~0.8
        """
        # 压力修正 under-relaxed亚松弛因子 即 p += alpha_p * p'
        self.alpha_p = 0.1
        # 速度修正 亚松弛因子
        self.alpha_u = 0.8
        # 速度更新 亚松弛因子
        self.alpha_m = 0.05

        # 速度场
        self.u = ti.field(dtype=float, shape=(nx+3, ny+2))
        self.v = ti.field(dtype=float, shape=(nx+2, ny+3))
        # 离散方程 迭代求解出的速度
        self.u_mid = ti.field(dtype=float, shape=(nx+3, ny+2))
        self.v_mid = ti.field(dtype=float, shape=(nx+2, ny+3))

        self.u0 = ti.field(dtype=float, shape=(
            nx+3, ny+2))  # Previous time step
        self.v0 = ti.field(dtype=float, shape=(nx+2, ny+3))

        # 压力场
        self.p = ti.field(dtype=float, shape=(nx+2, ny+2))
        # 压力修正值 p'
        self.pcor = ti.field(dtype=float, shape=(nx+2, ny+2))
        # 未使用
        self.pcor_mid = ti.field(dtype=float, shape=(nx+2, ny+2))
        # 连续性方程残差
        self.mdiv = ti.field(dtype=float, shape=(nx+2, ny+2))
        # 边界条件, 各边界的法向速度和切向速度
        self.bc = {'w': [0.0, 0.0], 'e': [0.0, 0.0],
                   'n': [0.0, 0.0], 's': [0.0, 0.0]}
        # unused
        self.ct = ti.field(dtype=float, shape=(nx+2, ny+2))   # Cell type

        # 分别对u、v控制容积积分, 标准离散格式代数方程下, 中心和四周 5个点的系数和偏置项
        # dim 1-w, 2-e, 3-n, 4-s
        self.coef_u = ti.field(dtype=float, shape=(nx+3, ny+2, 5))
        self.b_u = ti.field(dtype=float, shape=(nx+3, ny+2))
        self.coef_v = ti.field(dtype=float, shape=(nx+2, ny+3, 5))
        self.b_v = ti.field(dtype=float, shape=(nx+2, ny+3))
        # 对压力控制容积积分, 标准离散格式代数方程下, 中心和四周 5个点的系数和偏置项
        # dim 1-w, 2-e, 3-n, 4-s
        self.coef_p = ti.field(dtype=float, shape=(nx+2, ny+2, 5))
        self.b_p = ti.field(dtype=float, shape=(nx+2, ny+2))

        self.disp = Display(self, show_gui=True)

    @ti.kernel
    def compute_coef_u(self):
        """计算 u_p, u_N, u_S, u_W, u_E 的系数 和 b;
        离散格式: a_P·u_p + a_W·u_W + a_E·u_E + a_N·u_N + a_S·u_S = b
        """
        nx, ny, dx, dy, dt, rho, mu = self.nx, self.ny, self.dx, self.dy, self.dt, self.rho, self.mu
        for i, j in ti.ndrange((2, nx+1), (1, ny+1)):
            self.coef_u[i, j, 1] = -(mu * dy / dx + 0.5 * rho *
                                     0.5 * (self.u[i, j] + self.u[i-1, j]) * dy)      # aw
            self.coef_u[i, j, 2] = -(mu * dy / dx - 0.5 * rho *
                                     0.5 * (self.u[i, j] + self.u[i+1, j]) * dy)      # ae
            self.coef_u[i, j, 3] = -(mu * dx / dy - 0.5 * rho *
                                     0.5 * (self.v[i-1, j+1] + self.v[i, j+1]) * dx)  # an
            self.coef_u[i, j, 4] = -(mu * dx / dy + 0.5 * rho *
                                     0.5 * (self.v[i-1, j] + self.v[i, j]) * dx)      # as
            self.coef_u[i, j, 0] = -(self.coef_u[i, j, 1] + self.coef_u[i, j, 2] + self.coef_u[i, j, 3] +
                                     self.coef_u[i, j, 4]) +\
                rho * 0.5 * (self.u[i, j] + self.u[i+1, j]) * dy -\
                rho * 0.5 * (self.u[i, j] + self.u[i-1, j]) * dy +\
                rho * 0.5 * (self.v[i-1, j+1] + self.v[i, j+1]) * dx -\
                rho * 0.5 * (self.v[i-1, j] + self.v[i, j]) * dx +\
                rho * dx * dy / dt           # 非稳态项积分当前时刻的U_p的系数           # ap
            # 非稳态项的初值项u_p^0和压力梯度项 组成的b
            self.b_u[i, j] = (self.p[i-1, j] - self.p[i, j]) * dy + \
                rho * dx * dy / dt * self.u0[i, j]           # rhs

    @ti.kernel
    def compute_coef_v(self):
        """计算 v_P, v_N, v_S, v_W, v_E 的系数 和 b;
        离散格式: a_P·v_p + a_W·v_W + a_E·v_E + a_N·v_N + a_S·v_S = b
        """
        nx, ny, dx, dy, dt, rho, mu = self.nx, self.ny, self.dx, self.dy, self.dt, self.rho, self.mu
        for i, j in ti.ndrange((1, nx+1), (2, ny+1)):
            self.coef_v[i, j, 1] = -(mu * dy / dx + 0.5 * rho *
                                     0.5 * (self.u[i, j] + self.u[i, j-1]) * dy)       # aw
            self.coef_v[i, j, 2] = -(mu * dy / dx - 0.5 * rho *
                                     0.5 * (self.u[i+1, j-1] + self.u[i+1, j]) * dy)   # ae
            self.coef_v[i, j, 3] = -(mu * dx / dy - 0.5 * rho *
                                     0.5 * (self.v[i, j+1] + self.v[i, j]) * dx)       # an
            self.coef_v[i, j, 4] = -(mu * dx / dy + 0.5 * rho *
                                     0.5 * (self.v[i, j-1] + self.v[i, j]) * dx)       # as
            self.coef_v[i, j, 0] = -(self.coef_v[i, j, 1] + self.coef_v[i, j, 2] + self.coef_v[i, j, 3] +
                                     self.coef_v[i, j, 4]) +\
                rho * 0.5 * (self.u[i+1, j-1] + self.u[i+1, j]) * dy -\
                rho * 0.5 * (self.u[i, j] + self.u[i, j-1]) * dy +\
                rho * 0.5 * (self.v[i, j+1] + self.v[i, j]) * dx -\
                rho * 0.5 * (self.v[i, j-1] + self.v[i, j]) * dx +\
                rho * dx * dy / dt           # 非稳态项积分当前时刻的U_p的系数           # ap
            # 非稳态项的v_p^0和压力梯度项 组成的b
            self.b_v[i, j] = (self.p[i, j-1] - self.p[i, j]) * dx + \
                rho * dx * dy / dt * self.v0[i, j]           # rhs

    @ti.kernel
    def compute_mdiv(self) -> ti.f64:
        """计算不可压缩流连续性方程残差

        Returns:
            ti.f64: 各控制容积的剩余质量的绝对值最大值, 作为速度场迭代是否收敛的一个指标
        """
        nx, ny, dx, dy, rho = self.nx, self.ny, self.dx, self.dy, self.rho
        max_mdiv = 0.0
        ti.loop_config(serialize=True)
        for i, j in ti.ndrange((1, nx+1), (1, ny+1)):  # [1,nx], [1,ny]
            # b 项代表控制容积不能满足连续性的剩余质量的大小
            # [(\rho u)_w - (\rho u)_e] \cdot \Delta y + [(\rho v)_s - (\rho v)_n] \cdot \Delta x
            self.mdiv[i, j] = rho * (self.u[i, j] - self.u[i+1, j]) * \
                dy + rho * (self.v[i, j] - self.v[i, j+1]) * dx
            if ti.abs(self.mdiv[i, j]) > max_mdiv:
                max_mdiv = ti.abs(self.mdiv[i, j])
        return max_mdiv

    @ti.kernel
    def compute_coef_p(self):
        """压力修正方法, 压力控制容积的离散格式
        """
        nx, ny, dx, dy, rho = self.nx, self.ny, self.dx, self.dy, self.rho
        for i, j in ti.ndrange((1, nx+1), (1, ny+1)):  # [1,nx], [1,ny]
            # [(ρu)_w - (ρu)_e]· \Delta y + [(ρv)_s - (ρv)_n]· \Delta x
            self.mdiv[i, j] = rho * (self.u[i, j] - self.u[i+1, j]) * \
                dy + rho * (self.v[i, j] - self.v[i, j+1]) * dx
            self.b_p[i, j] = self.mdiv[i, j]
            # a_W = \rho_e {A_e / a_n} \Delta y = \rho \Delta y \Delta y / a_w
            self.coef_p[i, j, 1] = -rho * dy * dy / self.coef_u[i, j, 0]  # -aw
            self.coef_p[i, j, 2] = -rho * dy * \
                dy / self.coef_u[i+1, j, 0]  # -ae
            self.coef_p[i, j, 3] = -rho * dx * \
                dx / self.coef_v[i, j+1, 0]  # -an
            self.coef_p[i, j, 4] = -rho * dx * dx / self.coef_v[i, j, 0]  # -as

            # 边界上无论是法向速度已知还是压力分布已知, 都推导出相应影响系数为0
            if i == 1:
                # 左边界, a_W = 0
                self.coef_p[i, j, 1] = 0.0
            if i == nx:
                # 右边界, a_E = 0
                self.coef_p[i, j, 2] = 0.0
            if j == 1:
                # 南边界, a_S = 0
                self.coef_p[i, j, 4] = 0.0
            if j == ny:
                # 北边界, a_N = 0
                self.coef_p[i, j, 3] = 0.0
            # a_P = a_W + a_E + a_N + a_S
            self.coef_p[i, j, 0] = - (self.coef_p[i, j, 1] + self.coef_p[i, j, 2] +
                                      self.coef_p[i, j, 3] + self.coef_p[i, j, 4])
        # 压力参考点为左下角的内节点
        # a_P=1, a_E=a_W=a_N=a_S=b=0 设定压力参考点压力为0
        # 其他点的压力都是对于该点的压力的相对大小
        self.coef_p[1, 1, 1] = 0.0
        self.coef_p[1, 1, 2] = 0.0
        self.coef_p[1, 1, 3] = 0.0
        self.coef_p[1, 1, 4] = 0.0
        self.coef_p[1, 1, 0] = 1.0
        self.b_p[1, 1] = 0.0

    @ti.kernel
    def set_bc(self):
        """速度场边界条件设置
        """
        nx, ny, bc = self.nx, self.ny, self.bc
        # u - [nx+3, ny+2] - i E [0,nx+2], j E [0,ny+1]
        # v - [nx+2, ny+3] - i E [0,nx+1], j E [0,ny+2]
        # 边界处速度已知, 将边界速度项合进b项, 设相应边界影响系数为0
        for j in range(1, ny+1):
            # u bc for w
            self.b_u[2, j] += - self.coef_u[2, j, 1] * \
                bc['w'][0]       # b += aw * u_inlet
            self.coef_u[2, j, 1] = 0.0                                 # aw = 0
            # u[1, j]是p[1, j]的左边界速度, 即方形空腔左边界处速度
            self.u[1, j] = bc['w'][0]                                 # u_inlet

            # u bc for e
            self.b_u[nx, j] += - self.coef_u[nx, j, 2] * \
                bc['e'][0]     # b += ae * u_outlet
            self.coef_u[nx, j, 2] = 0.0                                # ae = 0
            # u_outlet u[nx+1, j]是p[nx+1, j]的右边界速度, 即方形空腔右边界处速度
            self.u[nx+1, j] = bc['e'][0]

        for i in range(1, nx+1):
            # v bc for s
            self.b_v[i, 2] += - self.coef_v[i, 2, 4] * \
                bc['s'][0]       # b += as * v_inlet
            self.coef_v[i, 2, 4] = 0.0                                 # as = 0
            self.v[i, 1] = bc['s'][0]                                 # v_inlet
            # v bc for n
            self.b_v[i, ny] += - self.coef_v[i, ny, 3] * \
                bc['n'][0]     # b += an * v_outlet
            self.coef_v[i, ny, 3] = 0.0                                # an = 0
            # v_outlet
            self.v[i, ny+1] = bc['n'][0]

        for i in range(2, nx+1):
            self.b_u[i, 1] += 2 * self.mu * bc['s'][1] * \
                self.dx / self.dy  # South sliding wall
            self.coef_u[i, 1, 0] += (self.coef_u[i, 1, 4] +
                                     2 * self.mu * self.dx / self.dy)
            self.coef_u[i, 1, 4] = 0.0
            # ap = ap - as + 2mudx/dy

            # ny处为与顶盖接触边界, 边界速度已知, 相应项直接计算归入b项
            self.b_u[i, ny] += 2 * self.mu * bc['n'][1] * \
                self.dx / self.dy  # North sliding wall
            # 并且边界速度不必做中心估计, 相应的aN为0, aP也少了aN的部分
            self.coef_u[i, ny, 0] += (self.coef_u[i,
                                      ny, 3] + 2 * self.mu * self.dx / self.dy)
            self.coef_u[i, ny, 3] = 0.0
            # ap = ap - an + 2mudx/dy

        for j in range(2, ny+1):
            self.b_v[1, j] += 2 * self.mu * bc['w'][1] * \
                self.dy / self.dx  # West sliding wall
            self.coef_v[1, j, 0] += (self.coef_v[1, j, 1] +
                                     2 * self.mu * self.dy / self.dx)
            self.coef_v[1, j, 1] = 0.0

            self.b_v[nx, j] += 2 * self.mu * bc['e'][1] * \
                self.dy / self.dx  # East sliding wall
            self.coef_v[nx, j, 0] += (self.coef_v[nx,
                                      j, 2] + 2 * self.mu * self.dy / self.dx)
            self.coef_v[nx, j, 2] = 0.0

    def bicg_solve_momentum_eqn(self, n_iter):
        """SIMPLE step3 - 动量守恒方程离散求解

        Args:
            n_iter (int): 内迭代次数?

        Returns:
            ti.f64: 动量守恒残差
        """
        residual = 0.0
        for i in range(n_iter):
            # 离散格式方程系数和偏置项
            self.compute_coef_u()
            self.compute_coef_v()
            self.set_bc()
            # self.u_momentum_solver.update_coef(
            #     self.coef_u, self.b_u, self.u_mid)
            # 迭代求解出速度u, 存在self.u_mid
            self.u_momentum_solver.solve(eps=1e-4, quiet=True)
            # self.v_momentum_solver.update_coef(
            #     self.coef_v, self.b_v, self.v_mid)
            # 迭代求解出速度v, 存在self.v_mid
            self.v_momentum_solver.solve(eps=1e-4, quiet=True)
            residual = self.update_velocity()
        return residual

    @ti.kernel
    def update_velocity(self) -> ti.f64:
        """对代数方程的速度解进行亚松驰存到self.u 、self.v 并计算动量方程残差"""
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        max_udiff = 0.0
        max_vdiff = 0.0
        ti.loop_config(serialize=True)
        for i, j in ti.ndrange((2, nx+1), (1, ny+1)):
            if ti.abs(self.u_mid[i, j] - self.u[i, j]) > max_udiff:
                max_udiff = ti.abs(self.u_mid[i, j] - self.u[i, j])
            # u^new = α_u·u^n + (1 - α_u)·u^(n-1),  u^(n-1)为上一次迭代的最终值, 即self.u
            # u^n 为本次迭代中动量方程的解
            self.u[i, j] = self.alpha_m * self.u_mid[i, j] + \
                (1 - self.alpha_m) * self.u[i, j]
        for i, j in ti.ndrange((1, nx+1), (2, ny+1)):
            if ti.abs(self.v_mid[i, j] - self.v[i, j]) > max_vdiff:
                max_vdiff = ti.abs(self.v_mid[i, j] - self.v[i, j])
            self.v[i, j] = self.alpha_m * self.v_mid[i, j] + \
                (1 - self.alpha_m) * self.v[i, j]
        return ti.sqrt(max_udiff ** 2 + max_vdiff ** 2)

    def bicg_solve_pcorrection_eqn(self, eps):
        """SIMPLE step4 - 压力修正方程求解

        Args:
            eps (_type_): _description_
        """
        self.compute_coef_p()
        # self.p_correction_solver.update_coef(self.coef_p, self.b_p, self.pcor)
        # 求解出p‘, 存在self.pcor
        self.p_correction_solver.solve(eps, quiet=True)

    @ti.kernel
    def correct_pressure(self):
        """SIMPLE step5 -压力修正"""
        nx, ny = self.nx, self.ny
        for i, j in ti.ndrange((1, nx+1), (1, ny+1)):
            self.p[i, j] += self.alpha_p * self.pcor[i, j]

    @ti.kernel
    def correct_velocity(self):
        """SIMPLE step5 -速度修正"""
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        for i, j in ti.ndrange((2, nx+1), (1, ny+1)):
            self.u[i, j] += self.alpha_u * \
                (self.pcor[i-1, j] - self.pcor[i, j]) * \
                dy / self.coef_u[i, j, 0]
        for i, j in ti.ndrange((1, nx+1), (2, ny+1)):
            self.v[i, j] += self.alpha_u * \
                (self.pcor[i, j-1] - self.pcor[i, j]) * \
                dx / self.coef_v[i, j, 0]

    @ti.kernel
    def init(self):
        """速度、压力场初始化为0
        """
        for i, j in self.u:
            self.u[i, j] = 0.0
            self.u0[i, j] = 0.0
        for i, j in self.v:
            self.v[i, j] = 0.0
            self.v0[i, j] = 0.0
        for i, j in self.p:
            self.p[i, j] = 0.0
            self.pcor[i, j] = 0.0

    def solve(self):
        """求解"""
        self.init()
        self.u_momentum_solver = BICGSolver(self.coef_u, self.b_u, self.u_mid)
        self.v_momentum_solver = BICGSolver(self.coef_v, self.b_v, self.v_mid)
        self.p_correction_solver = CGSolver(self.coef_p, self.b_p, self.pcor)
        momentum_residual = 0.0
        continuity_residual = 0.0
        for t in range(1):  # Time marching
            self.disp.matplt_display_init()
            # Internal iteration
            for substep in range(10000):
                # SIMPLE algorithm
                momentum_residual = self.bicg_solve_momentum_eqn(1)
                self.bicg_solve_pcorrection_eqn(1e-8)
                self.correct_pressure()
                self.correct_velocity()
                # 连续性方程的残差
                continuity_residual = self.compute_mdiv()
                # Printing residual to the prompt
                print(f'>>> Solving step {substep:06} Current continuity residual: {continuity_residual:.3e} \
                Current momentum residual: {momentum_residual:.3e}')
                self.disp.ti_gui_display(f'', show_gui=True)
                if substep % 10 == 1:
                    #self.disp.display(f'log/{substep:06}-corfin.png', show_gui=True)
                    self.disp.dump_field(substep, 'corfin')
                # Convergence check
                if momentum_residual < 1e-2 and continuity_residual < 1e-6:
                    print('>>> Solution converged.')
                    break
                #self.dump_coef(substep, 'momfin')
                self.disp.matplt_display_update(
                    substep, momentum_residual, continuity_residual)


# Lid-driven Cavity Setup
ssolver = SIMPLESolver(1.0, 1.0, 50, 50)  # lx, ly, nx, ny

# Boundary conditions
# ssolver.bc['w'][0] = 1.0    # West Normal velocity
# ssolver.bc['w'][1] = 1.0    # West Tangential velocity

# ssolver.bc['e'][0] = 1.0    # East Normal velocity
# ssolver.bc['e'][1] = 0.0    # East Tangential velocity

# ssolver.bc['n'][0] = 0.0    # North Normal velocity
# 空腔顶盖速度  北切向
ssolver.bc['n'][1] = 1.0    # North Tangential velocity

# ssolver.bc['s'][0] = 0.0    # South Normal velocity
# ssolver.bc['s'][1] = 0.0    # South Tangential velocity

ssolver.solve()
