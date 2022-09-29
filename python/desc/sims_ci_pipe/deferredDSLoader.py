"""
Wrapper class to defer loading of a dataset given its dataset ref.
"""

__all__ = ['DeferredDSLoader']

class DeferredDSLoader:
    """
    Wrapper class to defer loading of a dataset given its dataset ref.
    """
    def __init__(self, butler, dsref):
        self.butler = butler
        self.dsref = dsref

    def get(self, *args):
        if not args:
            return self.butler.get(self.dsref)
        else:
            return self.butler.get(args[0], dataId=self.dataId)

    @property
    def dataId(self):
        return self.dsref.dataId
