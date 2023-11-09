from .base import DictType


class Healthcheck(DictType):
    """
        Defines a healthcheck configuration for a container or service.

        Args:
            test (:py:class:`list` or str): Test to perform to determine
                container health. Possible values:

                - Empty list: Inherit healthcheck from parent image
                - ``["NONE"]``: Disable healthcheck
                - ``["CMD", args...]``: exec arguments directly.
                - ``["CMD-SHELL", command]``: Run command in the system's
                  default shell.

                If a string is provided, it will be used as a ``CMD-SHELL``
                command.
            interval (int): The time to wait between checks in nanoseconds. It
                should be 0 or at least 1000000 (1 ms).
            timeout (int): The time to wait before considering the check to
                have hung. It should be 0 or at least 1000000 (1 ms).
            retries (int): The number of consecutive failures needed to
                consider a container as unhealthy.
            start_period (int): Start period for the container to
                initialize before starting health-retries countdown in
                nanoseconds. It should be 0 or at least 1000000 (1 ms).
    """
    def __init__(self, **kwargs):
        test = kwargs.get('test', kwargs.get('Test'))
        if isinstance(test, str):
            test = ["CMD-SHELL", test]

        interval = kwargs.get('interval', kwargs.get('Interval'))
        timeout = kwargs.get('timeout', kwargs.get('Timeout'))
        retries = kwargs.get('retries', kwargs.get('Retries'))
        start_period = kwargs.get('start_period', kwargs.get('StartPeriod'))

        super().__init__({
            'Test': test,
            'Interval': interval,
            'Timeout': timeout,
            'Retries': retries,
            'StartPeriod': start_period
        })

    @property
    def test(self):
        return self['Test']

    @test.setter
    def test(self, value):
        if isinstance(value, str):
            value = ["CMD-SHELL", value]
        self['Test'] = value

    @property
    def interval(self):
        return self['Interval']

    @interval.setter
    def interval(self, value):
        self['Interval'] = value

    @property
    def timeout(self):
        return self['Timeout']

    @timeout.setter
    def timeout(self, value):
        self['Timeout'] = value

    @property
    def retries(self):
        return self['Retries']

    @retries.setter
    def retries(self, value):
        self['Retries'] = value

    @property
    def start_period(self):
        return self['StartPeriod']

    @start_period.setter
    def start_period(self, value):
        self['StartPeriod'] = value
