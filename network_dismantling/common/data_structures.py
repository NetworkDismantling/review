class dotdict(dict):
    """
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary

    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError
        return self.get(attr, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def product_dict(_callback=None, **kwargs):
    """
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists

    Args:
        _callback:
        **kwargs:

    Returns:

    """
    keys = kwargs.keys()
    vals = kwargs.values()

    from itertools import product

    for instance in product(*vals):
        instance = dict(zip(keys, instance))

        if _callback:
            instance = _callback(instance)

            if not instance:
                continue

        yield instance


class DefaultDict(dict):
    def __init__(self, value):
        super().__init__()
        self.__default_value = value

    def __missing__(self, key):
        return self.__default_value
