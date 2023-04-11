import sys
if 'setuptools' in sys.modules:
    from setuptools.command.sdist import sdist as old_sdist
else:
    from distutils.command.sdist import sdist as old_sdist

from numpy.distutils.misc_util import get_data_files

class sdist(old_sdist):

    def add_defaults (self):
        old_sdist.add_defaults(self)

        dist = self.distribution

        if dist.has_data_files():
            for data in dist.data_files:
                self.filelist.extend(get_data_files(data))

        if dist.has_headers():
            headers = []
            for h in dist.headers:
                if isinstance(h, str): headers.append(h)
                else: headers.append(h[1])
            self.filelist.extend(headers)

        return
