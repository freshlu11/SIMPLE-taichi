import taichi as ti

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


@ti.data_oriented
class Display:
    def __init__(self, SIMPLESolver, show_gui=False, *args):
        self.solver = SIMPLESolver
        self.nx = self.solver.nx
        self.ny = self.solver.ny

        self.udisp = ti.field(dtype=float, shape=((self.nx+2), (self.ny+2)))
        self.vdisp = ti.field(dtype=float, shape=((self.nx+2), (self.ny+2)))
        self.V_mag = ti.field(dtype=float, shape=((self.nx+2), (self.ny+2)))
        self.pdisp = ti.field(dtype=float, shape=((self.nx+2), (self.ny+2)))
        self.pcordisp = ti.field(dtype=float, shape=((self.nx+2), (self.ny+2)))
        self.mdivdisp = ti.field(dtype=float, shape=((self.nx+2), (self.ny+2)))
        self.gui = ti.GUI("SIMPLESolver", ((self.nx+2), 5 *
                          (self.ny+2)), show_gui=show_gui)

    @ti.func
    def scale_field(self, f):
        f_max = -1.0e9
        f_min = 1.0e9
        for i, j in f:
            if f[i, j] > f_max:
                f_max = f[i, j]
            if f[i, j] < f_min:
                f_min = f[i, j]
        for i, j in f:
            f[i, j] = (f[i, j] - f_min) / (f_max - f_min + 1.0e-9)

    @ti.kernel
    def post_process_field(self):
        for i, j in ti.ndrange(self.nx+2, self.ny+2):
            # 交错网格, 将速度场插值到压力场节点
            self.udisp[i, j] = 0.5 * \
                (self.solver.u[i, j] + self.solver.u[i+1, j])
            self.vdisp[i, j] = 0.5 * \
                (self.solver.v[i, j] + self.solver.v[i, j+1])
            self.V_mag[i, j] = ti.sqrt(self.udisp[i,j] ** 2 + self.vdisp[i,j] ** 2)
            # 压力场
            self.pdisp[i, j] = self.solver.p[i, j]
            self.pcordisp[i, j] = self.solver.pcor[i, j]
            # 压力控制容积-质量守恒
            self.mdivdisp[i, j] = self.solver.mdiv[i, j]
        # self.scale_field(self.udisp)
        # self.scale_field(self.vdisp)
        self.scale_field(self.pdisp)
        self.scale_field(self.pcordisp)
        self.scale_field(self.mdivdisp)

    def ti_gui_display(self, filename, show_gui=False):
        self.post_process_field()
        img = np.concatenate((self.udisp.to_numpy(), self.vdisp.to_numpy(), self.pdisp.to_numpy(),
                              self.pcordisp.to_numpy(), self.mdivdisp.to_numpy()), axis=1)
        self.gui.set_image(img)
        if show_gui:
            self.gui.show()
        else:
            self.gui.show(filename)

    def matplt_display_init(self):
        # Matplotlib live plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 4, figsize=(16, 6))
        self.x = []
        self.y1 = []
        self.y2 = []
        self.line1, = self.ax[0][0].plot(self.x, self.y1)
        self.line2, = self.ax[1][0].plot(self.x, self.y2)
        self.ax[0][0].set_xlabel('Iteration')
        self.ax[0][0].set_ylabel('Momentum residual')
        self.ax[1][0].set_xlabel('Iteration')
        self.ax[1][0].set_ylabel('Continuity residual')
        self.ax[0][0].grid()
        self.ax[1][0].grid()

        self.post_process_field()
        # self.vx, self.vy = self.udisp.shape
        self.vx = np.arange(0, 52, 1)
        self.vy = np.arange(0, 52, 1)
        self.VX, self.VY = np.meshgrid(self.vx, self.vy)
        self.ugraph = self.ax[0][2].imshow(self.udisp.to_numpy())
        self.uvecp = self.ax[0][2].quiver(
            self.VX, self.VY, self.udisp.to_numpy(), self.vdisp.to_numpy())

        self.ax[0][2].set_xlabel('U Velocity')
        self.ax[0][3].set_xlabel('U magnitude')
        self.ax[1][3].set_xlabel('U direction')
        self.vgraph = self.ax[1][2].imshow(self.vdisp.to_numpy())
        self.ax[1][2].set_xlabel('V Velocity')

        y_ref, u_ref = np.loadtxt(
            'data/ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 1))
        # Compare with Ghia's reference data
        self.ax[0][1].plot(y_ref, u_ref, 'cs', label='Ghia et al. 1982')
        self.u_xcor = np.linspace(0.01, 0.99, 50)
        self.u_ycor = self.solver.u.to_numpy()[26, 1:51]
        self.uprof, = self.ax[0][1].plot(
            self.u_xcor, self.u_ycor, label='Current u profile')
        self.ax[0][1].set_xlabel('U velocity profile at x = 0.5')
        self.ax[0][1].grid()
        self.ax[0][1].legend()

        x_ref, v_ref = np.loadtxt(
            'data/ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 7))
        # Compare with Ghia's reference data
        self.ax[1][1].plot(x_ref, v_ref, 'cs', label='Ghia et al. 1982')
        self.v_xcor = np.linspace(0.01, 0.99, 50)
        self.v_ycor = self.solver.v.to_numpy()[1:51, 26]
        self.vprof, = self.ax[1][1].plot(
            self.v_xcor, self.v_ycor, label='Current v profile')
        self.ax[1][1].set_xlabel('V velocity profile at y = 0.5')
        self.ax[1][1].grid()
        self.ax[1][1].legend()
        plt.tight_layout()

    def matplt_display_update(self, substep, momentum_residual, continuity_residual):
        # Update live plotting
        self.x.append(substep)
        self.y1.append(momentum_residual)
        self.y2.append(continuity_residual)
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(self.y1)
        self.line2.set_xdata(self.x)
        self.line2.set_ydata(self.y2)
        self.ax[0][0].relim()
        self.ax[0][0].autoscale_view()
        self.ax[1][0].relim()
        self.ax[1][0].autoscale_view()

        self.post_process_field()
        self.ax[0][2].cla()
        # 按模拟时的排布, 速度量的上下是方腔的东西方向, 左右是方腔南北方向
        u = np.flip((np.flip(self.udisp.to_numpy().transpose())), axis=1)
        v = np.flip((np.flip(self.vdisp.to_numpy().transpose())), axis=1)
        self.ax[0][2].imshow(u)
        # self.ugraph.set_data(np.flip(np.flip(self.udisp.to_numpy().transpose()), axis=1))
        # self.ax[0][2].quiver(self.VX, self.VY, np.flip(np.flip(self.udisp.to_numpy().transpose())),np.flip(np.flip(self.vdisp.to_numpy().transpose())))
        self.ax[1][3].quiver(self.VX, self.VY, u, v)
        # 速度量绘制
        V_np = np.flip((np.flip(self.V_mag.to_numpy().transpose())), axis=1)
        V_img = cm.jet(V_np)
        self.ax[0][3].imshow(V_img)  # Plot the velocity magnitude contour
        # self.gui.show()
        self.ugraph.autoscale()
        self.vgraph.set_data(v)
        self.vgraph.autoscale()
        self.uprof.set_ydata(self.solver.u.to_numpy()[26, 1:51])
        self.vprof.set_ydata(self.solver.v.to_numpy()[1:51, 26])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def dump_field(self, step, msg):  # Save u,v,p at step to csv files
        for name, val in {'u': self.solver.u, 'v': self.solver.v, 'p': self.solver.p,
                          'mdiv': self.solver.mdiv, 'pcor': self.solver.pcor}.items():
            np.savetxt(f'log/{step:06}-{name}-{msg}.csv',
                       val.to_numpy(), delimiter=',')

    def dump_coef(self, step, msg):
        np.savetxt(f'log/{step:06}-apu-{msg}.csv',
                   self.coef_u.to_numpy()[:, :, 0], delimiter=',')
        np.savetxt(f'log/{step:06}-awu-{msg}.csv',
                   self.coef_u.to_numpy()[:, :, 1], delimiter=',')
        np.savetxt(f'log/{step:06}-aeu-{msg}.csv',
                   self.coef_u.to_numpy()[:, :, 2], delimiter=',')
        np.savetxt(f'log/{step:06}-anu-{msg}.csv',
                   self.coef_u.to_numpy()[:, :, 3], delimiter=',')
        np.savetxt(f'log/{step:06}-asu-{msg}.csv',
                   self.coef_u.to_numpy()[:, :, 4], delimiter=',')
        np.savetxt(f'log/{step:06}-bu -{msg}.csv',
                   self.b_u.to_numpy(),           delimiter=',')

        np.savetxt(f'log/{step:06}-apv-{msg}.csv',
                   self.coef_v.to_numpy()[:, :, 0], delimiter=',')
        np.savetxt(f'log/{step:06}-awv-{msg}.csv',
                   self.coef_v.to_numpy()[:, :, 1], delimiter=',')
        np.savetxt(f'log/{step:06}-aev-{msg}.csv',
                   self.coef_v.to_numpy()[:, :, 2], delimiter=',')
        np.savetxt(f'log/{step:06}-anv-{msg}.csv',
                   self.coef_v.to_numpy()[:, :, 3], delimiter=',')
        np.savetxt(f'log/{step:06}-asv-{msg}.csv',
                   self.coef_v.to_numpy()[:, :, 4], delimiter=',')
        np.savetxt(f'log/{step:06}-bv -{msg}.csv',
                   self.b_v.to_numpy(),           delimiter=',')

        np.savetxt(f'log/{step:06}-app-{msg}.csv',
                   self.coef_p.to_numpy()[:, :, 0], delimiter=',')
        np.savetxt(f'log/{step:06}-awp-{msg}.csv',
                   self.coef_p.to_numpy()[:, :, 1], delimiter=',')
        np.savetxt(f'log/{step:06}-aep-{msg}.csv',
                   self.coef_p.to_numpy()[:, :, 2], delimiter=',')
        np.savetxt(f'log/{step:06}-anp-{msg}.csv',
                   self.coef_p.to_numpy()[:, :, 3], delimiter=',')
        np.savetxt(f'log/{step:06}-asp-{msg}.csv',
                   self.coef_p.to_numpy()[:, :, 4], delimiter=',')
        np.savetxt(f'log/{step:06}-bp -{msg}.csv',
                   self.b_p.to_numpy(),           delimiter=',')
