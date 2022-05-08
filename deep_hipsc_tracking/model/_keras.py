""" Mock out keras when imports fail """

# Standard lib
import sys
import warnings
import functools
from importlib import import_module

# 3rd party
try:
    import keras
except ImportError:
    warnings.warn('Cannot import keras', ImportWarning)
    keras = None

try:
    import vis
except ImportError:
    warnings.warn('Cannot import keras_vis', ImportWarning)
    vis = None

try:
    import tensorflow
except ImportError:
    warnings.warn('Cannot import tensorflow', ImportWarning)

# Constants

PACKAGES = {
    'keras': keras,
    'vis': vis,
    'tensorflow': tensorflow,
}

if sys.version_info[:1] < (3, 6):
    ModuleNotFoundError = ImportError

# General magic import function


def _import_package(*args, **kwargs):
    """ Import modules and things from a package """
    package_name = kwargs['package_name']

    if '.' in package_name:
        package_name, module_prefix = package_name.split('.', 1)
    else:
        module_prefix = None

    if PACKAGES[package_name] is None:
        if len(args) < 2:
            return None
        return (None for _ in args)

    module = kwargs.get('module')
    if module_prefix is not None:
        if module is None:
            module = module_prefix
        else:
            module = module_prefix + '.' + module

    modules = []
    for name in args:
        if module is not None:
            name = module + '.' + name
        try:
            modules.append(import_module('.' + name, package=package_name))
            continue
        except ModuleNotFoundError:
            if '.' not in name:
                raise
        modname, attr = name.rsplit('.', 1)
        modules.append(getattr(import_module('.' + modname, package=package_name), attr))
    if len(args) < 1:
        return None
    elif len(args) < 2:
        return modules[0]
    else:
        return modules


# Exposed magic import functions

_import_keras = functools.partial(_import_package, package_name='tensorflow.keras')
assert _import_keras is not None

_import_keras_vis = functools.partial(_import_package, package_name='vis')
assert _import_keras_vis is not None
