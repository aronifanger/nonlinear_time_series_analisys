import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d 

from matplotlib import animation
from matplotlib import cm
from IPython.display import HTML

def create_animation_3d(xyz, gif_name="./img/atractor.gif", figsize=(8, 8), interval=30, frames=200):
    fig = plt.figure(figsize=figsize)
    ax = axes3d.Axes3D(fig)
    def init():
        # Plot the surface.
        ax.plot3D(xyz[0], xyz[1], xyz[2], alpha=0.7, lw=0.3)
        return fig,

    def animate(i):
        # azimuth angle : 0 deg to 360 deg
        ax.view_init(elev=10, azim=i*1)
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)
    ani.save(gif_name, writer='imagemagick')
    plt.close(fig)
    
def create_animation_3d_scatter(xyz, gif_name="./img/atractor.gif", figsize=(8, 8), interval=30, frames=200):
    fig = plt.figure(figsize=figsize)
    ax = axes3d.Axes3D(fig)
    def init():
        # Plot the surface.
        ax.scatter(xyz[0], xyz[1], xyz[2], alpha=0.7, linewidths=0.01, s=1)
        return fig,

    def animate(i):
        # azimuth angle : 0 deg to 360 deg
        ax.view_init(elev=10, azim=i*1)
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)
    ani.save(gif_name, writer='imagemagick')
    plt.close(fig)

    
def plot_animation(gif_name="./img/atractor.gif"):
    return HTML('<img src="{}">'.format(gif_name))