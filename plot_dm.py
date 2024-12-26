import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colorbar as clrbar, colors as clrs
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from qutip import Qobj

""" My own version of PULSEE's plot dm function. Modified from PULSEE for Qbrain publication figures """

# I can't use `fig.tight_layout()`. Seems to be incompatible with what I'm doing.
def plot_complex_density_matrix(
        dm: Qobj,
        dm_theory: Qobj | None = None,
        many_spin_indexing: list | None = None,
        show: bool = True,
        phase_limits: list | np.ndarray | None = None,
        show_legend: bool = True,
        fig_dpi: int = 800,
        save_to: str = "",
        fig_size: tuple[float, float] | None = None,
        label_size: int = 16,
        label_qubit: bool = True,
        view_angle: tuple[float] = (45, -15),
        zlim: tuple[float, float] | None = None,
        add_shade= False
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generates a 3D histogram displaying the amplitude and phase (with colors)
    of the elements of the passed density matrix.

    Inspired by QuTiP 4.0's matrix_histogram_complex function.
    https://qutip.org

    Parameters
    ----------
    dm : Qobj
        Density matrix to be plotted.

    many_spin_indexing : None or list
        If not None, the density matrix dm is interpreted as the state of
        a many spins' system, and this parameter provides the list of the
        dimensions of the subspaces of the full Hilbert space related to the
        individual nuclei of the system.
        The ordering of the elements of many_spin_indexing should match that of
        the single spins' density matrices in their tensor product resulting in dm.

        For example, a system of [spin-1/2 x spin-1 x spin-3/2] will correspond to:
        many_spin_indexing = [2, 3, 4]

        Default value is None.

    show : bool
        When False, the graph constructed by the function will not be
        displayed.
        Default value is True.

    phase_limits : list/array of two floats
        The phase-axis (colorbar) limits [min, max]

    show_legend : bool
        Show the legend for the complex angle.

    fig_dpi : int
        Image quality of the figure when showing and saving. Useful for
        publications. Default set to very high value.

    save_to : str
        If this is not the empty string, the plotted graph will be saved to the
        path ('directory/filename') described by this string.

        Default value is the empty string.

    fig_size :  (float, float)
         Width, height in inches.
         Default value is the empty string.

    label_size : int
         Default is 6

    label_qubit : bool
        Whether to show the labels in the qubit convention:
        ex) |01> as opposed to |1/2, -1/2>.
        Default is False

    view_angle : (float, float)
         A tuple of (azimuthal, elevation) viewing angles for the 3D plot.
         Default is (45 deg, -15 deg)

    zlim : (int, int)
        The z axis limits of the plot.

    Action
    ------
    If show=True, draws a histogram on a 2-dimensional grid representing the
    density matrix, with phase sentivit data.

    Returns
    -------
    An object of the class matplotlib.figure.Figure and an object of the class
    matplotlib.axis.Axis representing the figure built up by the function.

    """
    if not isinstance(dm, Qobj):
        raise TypeError("The matrix must be an instance of Qobj!")

    if many_spin_indexing is None:
        many_spin_indexing = dm.dims[0]

    #assert np.array_equal(np.array(dm), dm.full())
    #dm = np.array(dm)
    dm=dm.full()
    n = dm.size

    # Set width of the vertical bars
    dx = dy = 0.7 * np.ones(n)
    dm_data = dm.flatten()
    dz = np.abs(dm_data)
    # Create an X-Y mesh of the same dimension as the 2D data. You can think of this as the floor of the plot
    xpos, ypos = np.meshgrid(range(dm.shape[0]), range(dm.shape[1]))
    xpos = xpos.T.flatten() - dx / 2
    ypos = ypos.T.flatten() - dy / 2
    zpos = np.zeros(n)

    # make small numbers real, to avoid random colors
    #idx, = np.where(abs(dm_data) < 0.001)
    #dm_data[idx] = abs(dm_data[idx])

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -np.pi
        phase_max = np.pi

    norm = clrs.Normalize(phase_min, phase_max)
    # cmap = pplt.Colormap('vikO', shift=-90)  # Using 'VikO' colormap from ProPlot
    # cmap = plt.get_cmap('twilight_shifted')
    cmap = rotate_colormap(plt.get_cmap('twilight'), angle=90, flip=True)
    colors = cmap(norm(np.angle(dm_data)))

    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure(constrained_layout=False)
    if fig_size:
        fig.set_size_inches(fig_size)
    ax = fig.add_subplot(111, projection="3d")

    if zlim is not None:
        ax.set_zlim(zlim)
    elif label_qubit:  # To display as figure in a paper.
        ax.set_zlim(0, 1)
        ax.set_zticks([0, 0.5, 1], [0, 0.5, 1], fontsize=label_size, verticalalignment='center')
    # Adjust the z tick label locations to they line up better with the ticks
    ax.tick_params('z', pad=0)

    # plot bars
    bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=add_shade, alpha=1, zsort='min')
    if not add_shade:
        bars.set_edgecolor('black')
        bars.set_linewidth(0.5)

    if dm_theory is not None:
        plot_dm_theory(ax, dm_theory, dx, dy, xpos, ypos, zpos)

    ax.view_init(elev=view_angle[0], azim=view_angle[1])  # rotating the plot so the "diagonal" direction is more clear

    """ Change I'm making for Qbrain publication """
    # turn off grid
    ax.grid(False)

    # turn off z-axis
    ax.zaxis.set_visible(False)
    ax.set_zticks([])
    ax.zaxis.set_tick_params(size=0)  # Removes tick labels
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Make axis line invisible
    ax.zaxis.pane.fill = False  # Disable z pane
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))  # Edge color invisible

    # turn off all the 'panes'
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    """ ######################################## """

    # Changed the code from PULSEE so `label_qubit` should always be true! Left it in for backwards compatibility (so
    # I don't have to go back to all the instances and get rid of all `label_qubit=True` arguments.
    if label_qubit:
        label_qubit_indices(ax, label_size, xpos, ypos)

    if show_legend:
        cax, kw = clrbar.make_axes(ax, location="right", shrink=0.7, pad=0)
        cb = clrbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels((r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"))
        cb.set_label("Phase")

    if save_to != "":
        plt.savefig(save_to, dpi=fig_dpi, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def plot_dm_theory(ax, dm_theory, dx, dy, xpos, ypos, zpos):
    dm_theory = dm_theory.full()
    dm_data_theo = dm_theory.flatten()
    dz_theo = np.abs(dm_data_theo)
    dx_theo = dy_theo = dx + 0.15
    xpos_theo = xpos + dx / 2 - dx_theo / 2
    ypos_theo = ypos + dy / 2 - dy_theo / 2
    # bars_theo = ax.bar3d(xpos_theo, ypos_theo, zpos, dx_theo, dy_theo, dz_theo,
    #          color=(0,0,0,0), shade=add_shade, zsort='min')
    # bars_theo.set_edgecolor('black')
    # bars_theo.set_linewidth(1)
    # Plot theoretical bars with vertical edges
    for i in range(len(xpos_theo)):
        vertices = generate_box_faces(
            xpos_theo[i], ypos_theo[i], zpos[i], dx_theo[i], dy_theo[i], dz_theo[i]
        )

        faces = Poly3DCollection(vertices, facecolor=(0, 0, 0, 0), edgecolor='black', alpha=0)
        ax.add_collection3d(faces)


def rotate_colormap(
        cmap: matplotlib.colors.Colormap,
        angle: float,
        flip: bool = False) -> matplotlib.colors.Colormap:
    """
    Helper function for `plot_complex_density_matrix`.

    Parameters
    ----------
    cmap: Colormap
        The colormap class to be shifted.
    angle: float
        IN DEGREES!
    flip: bool
        Whether to flip the color wheel. Note the flip is done AFTER the rotation.

    Returns
    -------
    a newly shifted colormap. Note that the flip is done AFTER the rotation.
    """
    n = 256
    nums = np.linspace(0, 1, n)
    shifted_nums = np.roll(nums, int(n * angle / 360))
    if flip:
        shifted_nums = np.flip(shifted_nums)
    shifted_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f"{cmap.name}_new", cmap(shifted_nums))
    return shifted_cmap


def label_qubit_indices(ax: plt.Axes, label_size: float, xpos, ypos):
    # convert the 16 coordinates into 4
    xpos = np.sort(np.unique(np.array(xpos))) + 0.25
    ypos = np.sort(np.unique(np.array(ypos))) + 0.25
    # adapted from qutip's `matrix_histogram_complex`
    labels = [r"$|$00$\rangle$", r"$|$01$\rangle$", r"$|$10$\rangle$", r"$|$11$\rangle$"]
    # ax.axes.xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    # ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', labelsize=label_size)
    #
    # ax.axes.yaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    # ax.set_yticklabels(labels)
    # ax.tick_params(axis='y', labelsize=label_size)

    ax.set_xticks(xpos, labels=labels, fontsize=label_size, va='bottom')
    ax.set_yticks(ypos, labels=labels, fontsize=label_size)
    ax.tick_params(axis='x', pad=10, color=(0, 0, 0, 0))
    ax.tick_params(axis='y', pad=-3, color=(0, 0, 0, 0))
    ax.tick_params(axis='z', direction='in', pad=0)


def label_qubit_indices_noline(ax: plt.Axes, label_size: float, xpos, ypos):
    # convert the 16 coordinates into 4
    xpos = np.sort(np.unique(np.array(xpos))) + 0.25
    ypos = np.sort(np.unique(np.array(ypos))) + 0.25
    # adapted from qutip's `matrix_histogram_complex`
    labels = [r"$|$00$\rangle$", r"$|$01$\rangle$", r"$|$10$\rangle$", r"$|$11$\rangle$"]
    # ax.axes.xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    # ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', labelsize=label_size)
    #
    # ax.axes.yaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    # ax.set_yticklabels(labels)
    # ax.tick_params(axis='y', labelsize=label_size)

    ax.set_xticks(xpos, labels=labels, fontsize=label_size, va='bottom')
    ax.set_yticks(ypos, labels=labels, fontsize=label_size)
    ax.tick_params(axis='x', pad=0, color=(0, 0, 0, 0))
    ax.tick_params(axis='y', pad=-10, color=(0, 0, 0, 0))
    ax.tick_params(axis='z', direction='in', pad=0)

    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Make axis line invisible
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Make axis line invisible


""" Helper only used for plotting the theoretical density matrix outline """


def generate_box_faces(x, y, z, dx, dy, dz):
    """Generates the six faces of a 3D box for given dimensions."""
    return [
        # Bottom
        [[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z]],
        # Top
        [[x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]],
        # Sides
        [[x, y, z], [x, y, z + dz], [x, y + dy, z + dz], [x, y + dy, z]],
        [[x + dx, y, z], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x + dx, y + dy, z]],
        [[x, y, z], [x + dx, y, z], [x + dx, y, z + dz], [x, y, z + dz]],
        [[x, y + dy, z], [x + dx, y + dy, z], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]],
    ]
