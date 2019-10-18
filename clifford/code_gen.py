from __future__ import print_function
import numpy as np


def generate_mult_function_batch_compile(mt, product_name, cuda=False):
    """
    Takes a given product and generates the code for a function that evaluates it
    """
    # Transpose to get the original memory order before sparse.COO changed it.
    # This doesn't really matter, but doing this prevents cuda_products.py changing.
    mt_T = mt.transpose((0, 2, 1))
    k_list, m_list, l_list = mt_T.coords
    mult_table_vals = mt_T.data
    n_dims = mt_T.shape[2]

    # Sort them by l list
    arg_list = np.argsort(l_list)

    def get_output_func_f_string(l_value):
        if cuda:
            f_string = '@cuda.jit(device=True)\n'
        else:
            f_string = '@njit\n'
        fname = product_name + '_o' + str(l_value)
        f_string += 'def ' + fname + '(value, other_value):\n'
        f_string += '    return 0'
        for ind in arg_list:
            l = l_list[ind]
            if l == l_value:
                k = k_list[ind]
                m = m_list[ind]
                mtv = mult_table_vals[ind]
                f_string += ' + ' + str(mtv) + '*value[' + str(k) + ']*other_value[' + str(m) + ']'

        return f_string

    total_string = ''
    if cuda:
        totalfuncstring = '@cuda.jit(device=True)\n'
    else:
        totalfuncstring = '@njit\n'
    totalfuncstring += 'def ' + product_name + '(value, other_value, output):\n'
    for i in range(n_dims):
        total_string += get_output_func_f_string(i) + '\n\n'
        f_name = product_name + '_o' + str(i)
        totalfuncstring += '    output[' + str(i) + '] = ' + f_name + '(value,other_value)\n'
    total_string += totalfuncstring
    return total_string


def write_mult_function_batch_compile(mt, product_name, file_obj, cuda=False):
    """
    Takes a given product and generates the code for a function that evaluates it, saves this to file
    """
    total_string = generate_mult_function_batch_compile(mt, product_name, cuda=cuda)
    print(total_string, file=file_obj)


def write_algebra(file_name, layout, cuda=False):
    """
    Writes the functions implementing gmt, omt and imt for a given layout into file_name
    """
    with open(file_name, 'w') as file_obj:
        # Write the preamble
        print('import numpy as np\nfrom numba import njit, cuda\n\n', file=file_obj)
        # Write the gmt
        write_mult_function_batch_compile(layout.gmt, 'gmt_func', file_obj, cuda=cuda)
        # Write the omt
        write_mult_function_batch_compile(layout.omt, 'omt_func', file_obj, cuda=cuda)
        # Write the imt
        write_mult_function_batch_compile(layout.imt, 'imt_func', file_obj, cuda=cuda)

if __name__ == '__main__':
    from clifford.g3c import *
    file_name = 'tools/g3c/cuda_products.py'
    write_algebra(file_name, layout, cuda=True)