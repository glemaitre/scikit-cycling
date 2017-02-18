"""
Helper to load some toy data.
"""
from os import listdir
from os.path import dirname
from os.path import join
from os.path import abspath


def load_toy(returned_type='list_file'):
    """Load some toy examples

    Parameters
    ----------
    returned_type : str, optional (default='list_file)
        If 'list_file', return a list containing the fit files;
        If 'path', return a string where the data are localized.

    Returns
    -------
    filenames : str or list of str,
        List of string or string depending of input parameters.
    """
    module_path = dirname(__file__)

    if returned_type == 'list_file':
        return [
            join(abspath(module_path), 'data', name)
            for name in listdir(join(abspath(module_path), 'data'))
            if name.endswith('.fit')
        ]
    elif returned_type == 'path':
        return join(abspath(module_path), 'data')
