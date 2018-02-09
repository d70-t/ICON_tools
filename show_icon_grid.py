import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def _main():
    import sys
    ds = xr.open_dataset(sys.argv[1])

    vertices = np.stack((ds.vlon.data, ds.vlat.data), axis=1)
    plt.scatter(vertices[:,0], vertices[:,1])
    plt.xlim(np.min(vertices[:,0]), np.max(vertices[:,0]))
    plt.ylim(np.min(vertices[:,1]), np.max(vertices[:,1]))
    edge_points = vertices[ds.edge_vertices.data-1]
    plt.plot(edge_points[...,0], edge_points[...,1], color="black", alpha=0.1)
    for i, vidx in enumerate(ds.vertex_of_cell.data.T):
        center = vertices[vidx-1].mean(axis=0)
        plt.text(center[0], center[1], "{:d}".format(i), horizontalalignment="center", verticalalignment="center")
    for i, eidx in enumerate(ds.edge_vertices.data.T):
        center = vertices[eidx-1].mean(axis=0)
        plt.text(center[0], center[1], "{:d}".format(i), horizontalalignment="center", verticalalignment="center", color="red")

    plt.show()


if __name__ == '__main__':
    _main()
