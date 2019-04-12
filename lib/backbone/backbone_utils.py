""" Wrapper functions for TensorFlow layers.
"""

import numpy as np
import tensorflow as tf

def expand_scope_by_name(scope, name):
    """ expand tf scope by given name.
    """

    if isinstance(scope, basestring):
        scope += '/' + name
        return scope

    if scope is not None:
        return scope.name + '/' + name
    else:
        return scope


def replicate_parameter_for_all_layers(parameter, n_layers):
    if parameter is not None and len(parameter) != n_layers:
        if len(parameter) != 1:
            raise ValueError()
        parameter = np.array(parameter)
        parameter = parameter.repeat(n_layers).tolist()
    return parameter


