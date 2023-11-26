from helpers.types_helper import Dataset, LatYLonXTimeIndices

NGSIM = Dataset(name='ngsim',
                hours_per_window=0.0833,
                first_window=1118846979,
                lat_y_min=0.0,
                lat_y_max=2235.252,
                lon_x_min=0.0,
                lon_x_max=75.313,
                resolution=(40, 40),
                indices=LatYLonXTimeIndices(1, 0, 3),
                is_lat_lon=False,
                paddingYX=(False, True))

SF = Dataset(name='sanfranciscocabs',
             hours_per_window=12,
             first_window=1210982400,
             last_window=1211403600,
             lat_y_min=37.71000,
             lat_y_max=37.81399,
             lon_x_min=-122.51584,
             lon_x_max=-122.38263,
             resolution=(300, 300),
             indices=LatYLonXTimeIndices(0, 1, 3))
