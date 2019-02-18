class sims_ci_pipe(object):
    '''
    Example class for sims_ci_pipe package.
    '''
    def __init__(self, message):
        self.message = message

    def run(self, raise_error=False):
        if raise_error:
            raise RuntimeError()
        return self.message
