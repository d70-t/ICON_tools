import numpy as np
import xarray as xr

RENUMBERING_FIELDS = dict([
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
    ])


def valid_cells(grid, cell_ids):
    valid = (cell_ids >= 0) & (cell_ids < grid.dims["cell"])
    return cell_ids[valid]

def valid_vertices(grid, vertex_ids):
    valid = (vertex_ids >= 0) & (vertex_ids < grid.dims["vertex"])
    return vertex_ids[valid]

def valid_edges(grid, edge_ids):
    valid = (edge_ids >= 0) & (edge_ids < grid.dims["edge"])
    return edge_ids[valid]

def grow_via_vertex(grid, vertex_of_cell, cells_of_vertex, cell_ids):
    vertices = valid_vertices(grid, np.unique(vertex_of_cell[cell_ids]))
    new_cells = valid_cells(grid, np.unique(cells_of_vertex[vertices]))
    return np.setdiff1d(new_cells, cell_ids)

def mk_inv_index(size, fwd_index):
    out = np.zeros(size+1, dtype="i4")-1
    out[fwd_index+1] = np.arange(1, len(fwd_index)+1, dtype="i4")
    return out

def cut_around_vertex(grid, center_vertex_id, radius):
    """
    :param grid: ICON grid as xarray Dataset
    :param center_vertex_id: (0-based) id of central vertex
    :param radius: radius of cells to export around vertex (in number of cells)
    :return: (new_grid, renumbering_table)
    """

    assert(len(valid_vertices(grid, np.array([center_vertex_id]))) > 0)
    vertex_of_cell = grid.vertex_of_cell.load().transpose("cell", "nv").data - 1
    cells_of_vertex = grid.cells_of_vertex.load().transpose("vertex", "ne").data - 1
    cells = valid_cells(grid, cells_of_vertex[center_vertex_id])

    if radius >= 0:
        new_cells = cells
        while radius > 0:
            print("radius:", radius)
            new_cells = np.setdiff1d(grow_via_vertex(grid, vertex_of_cell, cells_of_vertex, new_cells), cells)
            cells = np.union1d(cells, new_cells)
            radius -= 1
    else:
        cells = cells[:1]

    print("searching vertices")
    vertices = valid_vertices(grid, np.unique(vertex_of_cell[cells]))
    print("searching edges")
    edges = valid_edges(grid, np.unique(grid.edge_of_cell.load().isel(cell=cells)) - 1)

    print("generating renumbering tables")
    renumbering_table = xr.Dataset({
        "cell_renumbering": xr.DataArray(cells+1, dims=("cell",)),
        "edge_renumbering": xr.DataArray(edges+1, dims=("edge",)),
        "vertex_renumbering": xr.DataArray(vertices+1, dims=("vertex",)),
        })

    print("generating inverse indices")
    inv_indices = {
            "cell": mk_inv_index(grid.dims["cell"], cells),
            "edge": mk_inv_index(grid.dims["edge"], edges),
            "vertex": mk_inv_index(grid.dims["vertex"], vertices),
        }
    slices = {"cell": cells, "edge": edges, "vertex": vertices}

    out_vars = {}
    for name, var in grid.variables.items():
        print("converting variable {}".format(name))
        renumber_field = RENUMBERING_FIELDS.get(name, None)
        local_slices = {k:v for k, v in slices.items() if k in var.dims}
        temp = var.data
        try:
            temp = temp.compute()
        except AttributeError:
            pass
        temp = xr.DataArray(temp, dims=var.dims, attrs=var.attrs)
        out_vars[name] = temp.isel(**local_slices)
        if renumber_field is not None:
            out_vars[name].data = inv_indices[renumber_field][out_vars[name].data].astype(temp.dtype)
        del temp

    new_grid = xr.Dataset(out_vars, attrs=grid.attrs)
    return new_grid, renumbering_table

def find_central_vertex(grid, vertex_spec):
    try:
        return int(vertex_spec)
    except ValueError:
        parts = vertex_spec.split(",")
        if len(parts) != 2:
            raise ValueError("invalid vertex definition: {}".format(vertex_spec))
        lat, lon = [np.deg2rad(float(x)) for x in parts]
        distance2 = (grid.vlat.data - lat) ** 2 + (grid.vlon.data - lon) ** 2
        return np.argmin(distance2)


def _main():
    import argparse

    parser = argparse.ArgumentParser("ICON grid cutout tool")
    parser.add_argument("input_grid")
    parser.add_argument("output_grid")
    parser.add_argument("output_renumbering_table")
    parser.add_argument("central_vertex", help="either int, meaning a vertex index or <lat>,<lon> (comma separated latitude longitude pair in decimal degree notation) meaning the vertex closes to this position")
    parser.add_argument("radius", type=int, help="number of rings around central ring of cells, negative-> only one cell")

    args = parser.parse_args()

    grid = xr.open_dataset(args.input_grid)

    central_vertex = find_central_vertex(grid, args.central_vertex)
    print("cutting around vertex {}".format(central_vertex))
    new_grid, renumbering_table = cut_around_vertex(grid, central_vertex, args.radius)

    new_grid.to_netcdf(args.output_grid)
    renumbering_table.to_netcdf(args.output_renumbering_table)

    #print(new_grid)
    #print(renumbering_table)

if __name__ == '__main__':
    _main()
