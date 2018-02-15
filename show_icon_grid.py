import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def show_icon_grid(ds, show_vertex_numbers=False):
    vertices = np.stack((ds.vlon.data, ds.vlat.data), axis=1)
    plt.xlim(np.min(vertices[:,0]), np.max(vertices[:,0]))
    plt.ylim(np.min(vertices[:,1]), np.max(vertices[:,1]))
    edge_points = vertices[ds.edge_vertices.data-1]
    plt.plot(edge_points[...,0], edge_points[...,1], color="black", alpha=0.1)

    if show_vertex_numbers:
        for i, vertex in enumerate(vertices):
            plt.text(vertex[0], vertex[1], "{:d}".format(i), horizontalalignment="center", verticalalignment="center", color="C0")
    else:
        plt.scatter(vertices[:,0], vertices[:,1])
    for i, vidx in enumerate(ds.vertex_of_cell.data.T):
        center = vertices[vidx-1].mean(axis=0)
        plt.text(center[0], center[1], "{:d}".format(i), horizontalalignment="center", verticalalignment="center")
    for i, eidx in enumerate(ds.edge_vertices.data.T):
        center = vertices[eidx-1].mean(axis=0)
        plt.text(center[0], center[1], "{:d}".format(i), horizontalalignment="center", verticalalignment="center", color="red")

    plt.show()

def _main():
    import argparse

    parser = argparse.ArgumentParser("icon grid display")
    parser.add_argument("gridfile", help="icon grid file")
    parser.add_argument("-v", "--vertices", help="show vertex numbers", default=False, action="store_true")

    args = parser.parse_args()
    ds = xr.open_dataset(args.gridfile)
    show_icon_grid(ds, args.vertices)

if __name__ == '__main__':
    _main()
