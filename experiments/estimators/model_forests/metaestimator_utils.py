class AttributeWrapperMixin:
    """A mixin that allows accessing attributes of the wrapped estimator.

    If an attribute is not found in the current class,
    it will be searched in the meta attributes.
    """

    _meta_attributes = ("estimator_", "estimator")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                for meta_attr in self._meta_attributes:
                    return getattr(getattr(self, meta_attr), name)
            except AttributeError:
                pass

        type_dict = {
            attr: (
                getattr(self, attr).__class__.__name__
                if hasattr(self, attr)
                else "(Not set!)"
            )
            for attr in self._meta_attributes
        }
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
            f" and nor does its attribute(s): {type_dict}"
        )
