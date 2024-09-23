class MockObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def __call__(self, *args, **kwargs):
        return self

    def __setattr__(self, name, value):
        self.__dict__[name] = value
