def set_module(module_name):
    """
    Helper to override the `__module__` of a definition.

    This is useful for moving class and function definitions to subfiles
    without affecting:
    * :mod:`pickle`, which will saves the module name
    * Sphinx, which uses the module name in link targets
    """
    def decorator(obj):
        obj.__module__ = module_name
        return obj
    return decorator
