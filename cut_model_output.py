import numpy as np
import itertools
import netCDF4

class ChunkCopyHelper(object):
    def __init__(self, table, chunksize_hint=1000, source_size=None):
        self.table = np.sort(table)
        ncount = int(np.max(self.table) / chunksize_hint) + 1
        is_active = np.zeros(ncount, dtype="bool")
        is_active[self.table / chunksize_hint] = True
        groups = [(idx[0][1], idx[-1][1]) for idx in (list(idx) for k, idx in itertools.groupby(zip(is_active, itertools.count(0)), lambda x: x[0]) if k)]
        group_bounds = [(lo * chunksize_hint, (hi+1) * chunksize_hint) for lo, hi in groups]
        group_indices = [self.table[(self.table>=lo) & (self.table<hi)] for lo, hi in group_bounds]
        self.sources = []
        copy_count = 0
        for (lo, hi), idx in zip(group_bounds, group_indices):
            lo = idx[0]
            hi = idx[-1]+1
            idx -= idx[0]
            self.sources.append((slice(lo, hi), idx))
            copy_count += hi - lo
        gcount = map(len, group_indices)
        target_borders = np.concatenate([[0], np.cumsum(gcount)])
        self.target_slices = []
        for a, b in zip(target_borders[:-1], target_borders[1:]):
            self.target_slices.append(slice(a, b))

        if source_size is None:
            source_size = self.table[-1] + 1
        print("chunk copy helper, estimated reduction: {:.1f}%".format(100.*copy_count/source_size))

    def copy(self, source_var, target_var, slices_before=(), slices_after=()):
        for (ssl, sidx), t in zip(self.sources, self.target_slices):
            target_idx = slices_before + (t,) + slices_after
            source_prefetch_idx = slices_before + (ssl,) + slices_after
            target_var[target_idx] = np.array(source_var[source_prefetch_idx])[sidx]


def cut_model_data(data, renumbering_table, target):
    out_cells = len(renumbering_table)

    print data.dimensions
    for dim in data.dimensions.values():
        if dim.name == "ncells":
            size = out_cells
        else:
            size = dim.size
        target.createDimension(dim.name, size)

    copyhelper = ChunkCopyHelper(renumbering_table, chunksize_hint=10000, source_size = data.dimensions["ncells"].size)

    for name, var in data.variables.items():
        print(name)
        tvar = target.createVariable(name, var.dtype, var.dimensions)
        if "ncells" not in var.dimensions:
            tvar = var
            continue
        shape = var.shape
        cell_index = var.dimensions.index("ncells")
        copy_shape = [s for i, s in enumerate(shape) if i != cell_index]
        for idx in itertools.product(*map(range, copy_shape)):
            copyhelper.copy(var, tvar, idx[:cell_index], idx[cell_index:])

def _main():
    import argparse

    parser = argparse.ArgumentParser("ICON model output cutout tool")
    parser.add_argument("input_model_data")
    parser.add_argument("renumbering_table", help="this can be generatred using cut_grid.py")
    parser.add_argument("output_model_data")

    args = parser.parse_args()

    data = netCDF4.Dataset(args.input_model_data)
    renumbering_table = np.array(netCDF4.Dataset(args.renumbering_table).variables["cell_renumbering"]) - 1
    target = netCDF4.Dataset(args.output_model_data, "w")

    cut_model_data(data, renumbering_table, target)

if __name__ == '__main__':
    _main()
