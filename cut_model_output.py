import xarray as xr

def cut_model_data(data, renumbering_table):
    return data.isel(ncells=renumbering_table.cell_renumbering.data-1)

def _main():
    import argparse

    parser = argparse.ArgumentParser("ICON model output cutout tool")
    parser.add_argument("input_model_data")
    parser.add_argument("renumbering_table", help="this can be generatred using cut_grid.py")
    parser.add_argument("output_model_data")

    args = parser.parse_args()

    data = xr.open_dataset(args.input_model_data)
    renumbering_table = xr.open_dataset(args.renumbering_table)

    cut_model_data(data, renumbering_table).to_netcdf(args.output_model_data)

if __name__ == '__main__':
    _main()
