# Using autodoc in Sphinx
See [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc) for more information

## Steps
1. Follow the coding style as depicted in [reStructuredText](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html). To save you some time, here's a template for the coding style:

```
def myfunc(arg1, arg2):
    """
    Facts about the myfunc. (Make sure to have a new line after the
    docstring and no separation between parameters and return value.)

    :param arg1: Argument #1 for myfunc.
    :type arg1: str
    :parar arg2: Argument #2 for myfunc.
    :type arg2: int
    :returns: The result value for myfunc(arg1, arg2)
    :rtype: float
    """

    # Do some work here

    return result
```

2. Build the files
```
sphinx-build -b html source build && make html
```

    Ignore the warning
```
.../docs/source/modules.rst: WARNING: document isn't included in any toctree
```

3. Verify the changes in the compiled html files under `docs/build/html/`. You can just open the browser to check it.
