import os
from distutils.command.install_headers import install_headers as old_install_headers

class install_headers (old_install_headers):

    def run (self):
        headers = self.distribution.headers
        if not headers:
            return

        prefix = os.path.dirname(self.install_dir)
        for header in headers:
            if isinstance(header, tuple):
                # Kind of a hack, but I don't know where else to change this...
                if header[0] == 'numpy.core':
                    header = ('numpy', header[1])
                    if os.path.splitext(header[1])[1] == '.inc':
                        continue
                d = os.path.join(*([prefix]+header[0].split('.')))
                header = header[1]
            else:
                d = self.install_dir
            self.mkpath(d)
            (out, _) = self.copy_file(header, d)
            self.outfiles.append(out)
