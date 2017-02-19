"""
Helper to load some toy data.
"""
from os import listdir
from os.path import dirname
from os.path import join
from os.path import abspath


def load_toy(returned_type='list_file', set_data='normal'):
    """Load some toy examples

    Parameters
    ----------
    returned_type : str, optional (default='list_file)
        If 'list_file', return a list containing the fit files;
        If 'path', return a string where the data are localized.

    set_data : str, optional (default='normal')
        If 'normal', return 3 files.
        If 'corrupted, return corrupted files for testing.

    Returns
    -------
    filenames : str or list of str,
        List of string or string depending of input parameters.
    """
    module_path = dirname(__file__)

    if set_data == 'normal':
        if returned_type == 'list_file':
            return sorted([
                join(abspath(module_path), 'data', name)
                for name in listdir(join(abspath(module_path), 'data'))
                if name.endswith('.fit')
            ])
        elif returned_type == 'path':
            return join(abspath(module_path), 'data')
    elif set_data == 'corrupted':
        if returned_type == 'list_file':
            return sorted([
                join(abspath(module_path), 'corrupted_data', name)
                for name in listdir(
                    join(abspath(module_path), 'corrupted_data'))
                if name.endswith('.fit')
            ])
        elif returned_type == 'path':
            return join(abspath(module_path), 'corrupted_data')
