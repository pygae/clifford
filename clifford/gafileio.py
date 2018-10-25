
import h5py
import numpy as np


def write_ga_file(file_name, mv_array, metric, basis_names, compression=True,
                   transpose=False, sparse=False, support=False, compression_opts=1):
    """
    Writes a ga file of format version 0.0.1
    """
    with h5py.File(file_name, "w") as f:
        # Record the version number
        f.attrs['version'] = '0.0.1'

        # First lets deal with the multivector coefficient data itself
        if compression:
            if transpose:
                dset_data = f.create_dataset("data", data=mv_array.T, compression="gzip",
                                             compression_opts=compression_opts)
                dset_data.attrs['transpose'] = True
            else:
                dset_data = f.create_dataset("data", data=mv_array, compression="gzip",
                                             compression_opts=compression_opts)
                dset_data.attrs['transpose'] = False
        else:
            if transpose:
                dset_data = f.create_dataset("data", data=mv_array.T)
                dset_data.attrs['transpose'] = True
            else:
                dset_data = f.create_dataset("data", data=mv_array)
                dset_data.attrs['transpose'] = False

        if sparse:
            dset_data.attrs['sparse'] = True
            if support is not None:
                dset_support = f.create_dataset("support", support)
            else:
                raise ValueError('You must specify the support of the multivectors '
                                 'if you explicitly specify sparse storage')
        else:
            dset_data.attrs['sparse'] = False
            dset_support = f.create_dataset("support", data=np.array([], dtype=np.uint64))

        # Now save the metric
        dset_metric = f.create_dataset("metric", data=metric)

        # Now the basis names
        dset_basis_names = f.create_dataset("basis_names", data=basis_names)


def read_ga_file(file_name):
    """
    Reads a ga file of format version 0.0.1
    """
    with h5py.File(file_name, "r") as f:
        assert f.attrs['version'] == '0.0.1'
        data = f['data']
        transpose = data.attrs['transpose']
        if transpose:
            data_array = data[:].T
        else:
            data_array = data[:]
        sparse = data.attrs['sparse']
        if sparse:
            support = data.attrs['support']
        else:
            support = None
        metric = f['metric'][:]
        basis_names = f['basis_names'][:]
    return data_array, metric, basis_names, support
