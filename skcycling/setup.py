def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('skcycling', parent_package, top_path)

    config.add_subpackage('data_management')
    config.add_subpackage('data_management/tests')
    config.add_subpackage('datasets')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('metrics')
    config.add_subpackage('metrics/tests')
    config.add_subpackage('power_profile')
    config.add_subpackage('power_profile/tests')
    config.add_subpackage('probabilistic_power_profile')
    config.add_subpackage('probabilistic_power_profile/tests')
    config.add_subpackage('restoration')
    config.add_subpackage('restoration/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')

    # external libraries
    config.add_subpackage('externals')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)
