import six


class Profiler(object):
    @classmethod
    def profile(cls, func):
        def do(*args, **kwargs):
            with cls():
                return func(*args, **kwargs)

        return do

    def __init__(self, enabled=True, sort='tottime'):
        self.enabled = enabled
        self._profiler = None
        self._sort = sort

    def __enter__(self):
        if not self.enabled:
            return

        import cProfile  # keep import nested - this module is not necessary in production
        self._profiler = cProfile.Profile()
        self._profiler.enable()

        return self._profiler

    def __exit__(self, *args):
        if self._profiler is None:
            return

        self._profiler.disable()

        buffer = six.StringIO()

        import pstats  # keep import nested - this module is not necessary in production
        pstats.Stats(self._profiler, stream=buffer) \
            .sort_stats(self._sort) \
            .print_callees(.25) \
            .print_stats(20) \

        print(buffer.getvalue())

        self._profiler = None