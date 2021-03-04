
import h5py
import numpy as np
import json


def write_json_file(file_name, mv_array, metric, basis_names, compression=True,
                    transpose=False, sparse=False, support=None, compression_opts=1):
    """
    Writes a json ga file of format version 0.0.1
    """
    data_dict = {}

    # Record the version number
    data_dict['version'] = '0.0.1'

    # First lets deal with the multivector coefficient data itself
    dset_data = {}
    if transpose:
        dset_data['data'] = mv_array.T.tolist()
        dset_data['transpose'] = True
    else:
        dset_data['data'] = mv_array.tolist()
        dset_data['transpose'] = False

    if sparse:
        dset_data['sparse'] = True
        if support is not None:
            dset_data['support'] = support.tolist()
        else:
            raise ValueError('You must specify the support of the multivectors '
                             'if you explicitly specify sparse storage')
    else:
        dset_data['sparse'] = False
        dset_data['support'] = []

    data_dict['dataset'] = dset_data

    # Now save the metric
    data_dict["metric"] = metric.tolist()

    # Now the basis names
    data_dict["basis_names"] = [str(s) for s in basis_names]

    with open(file_name, "w") as fp:
        json.dump(data_dict, fp)


def read_json_file(file_name):
    """
    Reads a json ga file of format version 0.0.1
    """
    with open(file_name, "r") as fp:
        f = json.load(fp)
        assert f['version'] == '0.0.1'
        data = f['dataset']
        transpose = data['transpose']
        if transpose:
            data_array = np.array(data['data']).T
        else:
            data_array = np.array(data['data'])
        sparse = data['sparse']
        if sparse:
            support = data['support']
        else:
            support = None
        metric = np.array(f['metric'])
        basis_names = np.array(f['basis_names'][:])
    return data_array, metric, basis_names, support


def write_ga_file(file_name, mv_array, metric, basis_names, compression=True,
                  transpose=False, sparse=False, support=None, compression_opts=1):
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
        try:
            dt = h5py.string_dtype()  # new in 2.10
        except AttributeError:
            dt = h5py.special_dtype(vlen=str)
        dset_basis_names = f.create_dataset("basis_names", data=np.asarray(basis_names, dtype=dt))


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
        if hasattr(h5py.Dataset, 'asstr'):
            basis_names = f['basis_names'].asstr()[:]
        else:
            # h5py < 3
            basis_names = f['basis_names'][:]
    return data_array, metric, basis_names, support
