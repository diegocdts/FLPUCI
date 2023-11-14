from helpers.types_helper import Dataset, LatYLonXTimeIndices

NGSIM = Dataset(name='ngsim',
                hours_per_window=0.1,
                first_window=1113433135300,
                lat_y_min=0.0,
                lat_y_max=1790.771,
                lon_x_min=0.0,
                lon_x_max=96.996,
                resolution=100,
                indices=LatYLonXTimeIndices(1, 0, 3),
                is_lat_lon=False)

SF = Dataset(name='sanfranciscocabs',
             hours_per_window=12,
             first_window=1210982400,
             lat_y_min=37.71000,
             lat_y_max=37.81399,
             lon_x_min=-122.51584,
             lon_x_max=-122.38263,
             resolution=300,
             indices=LatYLonXTimeIndices(0, 1, 3))
