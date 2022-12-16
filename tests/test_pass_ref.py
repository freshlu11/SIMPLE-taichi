import taichi as ti


ti.init(arch=ti.cpu, default_fp=ti.f64)


@ti.data_oriented
class CGSolver:
    """BICG算法求解 coef·x = b
    """
    def __init__(self, coef, b, x):
        self.coef = coef
        self.b = b
        self.x = x
        self.x1 = ti.field(dtype=float, shape=(x.shape[0], x.shape[1]))

    def solve(self):
        self.x[0, 0] = 1
        self.x1[0, 1] = self.x[0, 1]
        self.x1[0, 1] = 9


@ti.data_oriented
class SIMPLESolver:
    def __init__(self, lx, ly, nx, ny):
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.pcor = ti.field(dtype=float, shape=(nx+2, ny+2))
        self.coef_p = ti.field(dtype=float, shape=(nx+2, ny+2, 5))
        self.b_p = ti.field(dtype=float, shape=(nx+2, ny+2))

    @ti.kernel
    def init(self):
        """速度、压力场初始化为0
        """
        for i, j in self.pcor:
            self.pcor[i, j] = 0.0

    def solve(self):
        """求解"""
        self.init()
        self.p_correction_solver = CGSolver(self.coef_p, self.b_p, self.pcor)
        self.p_correction_solver.solve()
        print(self.pcor)


ssolver = SIMPLESolver(1.0, 1.0, 5, 5)  # lx, ly, nx, ny
ssolver.solve()
