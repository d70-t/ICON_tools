import numpy as np
import xarray as xr

RENUMBERING_FIELDS = [
    ("edge_of_cell", "edge"),
    ("vertex_of_cell", "vertex"),
    ("adjacent_cell_of_edge", "cell"),
    ("edge_vertices", "vertex"),
    ("cells_of_vertex", "cell"),
    ("edges_of_vertex", "edge"),
    ("vertices_of_vertex", "vertex"),

    ("neighbor_cell_index", "cell"),
    ("cell_index", "cell"),
    ("vertex_index", "vertex"),
    ("edge_index", "edge"),
    ]


def valid_cells(grid, cell_ids):
    valid = (cell_ids >= 0) & (cell_ids < grid.dims["cell"])
    return cell_ids[valid]

def valid_vertices(grid, vertex_ids):
    valid = (vertex_ids >= 0) & (vertex_ids < grid.dims["vertex"])
    return vertex_ids[valid]

def valid_edges(grid, edge_ids):
    valid = (edge_ids >= 0) & (edge_ids < grid.dims["edge"])
    return edge_ids[valid]

def grow_via_vertex(grid, cell_ids):
    vertices = valid_vertices(grid, np.unique(grid.vertex_of_cell.sel(cell=cell_ids).data) - 1)
    new_cells = valid_cells(grid, np.unique(grid.cells_of_vertex.sel(vertex=vertices).data) - 1)
    return np.setdiff1d(new_cells, cell_ids)

def cut_around_vertex(grid, center_vertex_id, radius):
    """
    :param grid: ICON grid as xarray Dataset
    :param center_vertex_id: (0-based) id of central vertex
    :param radius: radius of cells to export around vertex (in number of cells)
    :return: (new_grid, renumbering_table)
    """

    assert(len(valid_vertices(grid, np.array([center_vertex_id]))) > 0)
    cells = valid_cells(grid, grid.cells_of_vertex.sel(vertex=center_vertex_id).data - 1)
    new_cells = cells
    while radius > 0:
        new_cells = np.setdiff1d(grow_via_vertex(grid, new_cells), cells)
        cells = np.union1d(cells, new_cells)
        radius -= 1

    vertices = valid_vertices(grid, np.unique(grid.vertex_of_cell.sel(cell=cells)) - 1)
    edges = valid_edges(grid, np.unique(grid.edge_of_cell.sel(cell=cells)) - 1)

    renumbering_table = xr.Dataset({
        "cell_renumbering": xr.DataArray(cells+1, dims=("cell",)),
        "edge_renumbering": xr.DataArray(edges+1, dims=("edge",)),
        "vertex_renumbering": xr.DataArray(vertices+1, dims=("vertex",)),
        })

    new_grid = grid.sel(cell=cells, edge=edges, vertex=vertices)
    inv_indices = {
            "cell": {v+1:idx+1 for idx, v in enumerate(cells)},
            "edge": {v+1:idx+1 for idx, v in enumerate(edges)},
            "vertex": {v+1:idx+1 for idx, v in enumerate(vertices)},
        }

    for field, key in RENUMBERING_FIELDS:
        inv_index = inv_indices[key]
        remap = np.vectorize(lambda x: inv_index.get(x, -1))
        new_grid[field].data = remap(new_grid[field].data)

    return new_grid, renumbering_table

def _main():
    import argparse

    parser = argparse.ArgumentParser("ICON grid cutout tool")
    parser.add_argument("input_grid")
    parser.add_argument("output_grid")
    parser.add_argument("output_renumbering_table")
    parser.add_argument("central_vertex", type=int)
    parser.add_argument("radius", type=int)

    args = parser.parse_args()

    grid = xr.open_dataset(args.input_grid)

    new_grid, renumbering_table = cut_around_vertex(grid, args.central_vertex, args.radius)

    new_grid.to_netcdf(args.output_grid)
    renumbering_table.to_netcdf(args.output_renumbering_table)

    print(new_grid)
    print(renumbering_table)

if __name__ == '__main__':
    _main()
