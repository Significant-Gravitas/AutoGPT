import io
import os
import re
import tarfile
import tempfile

from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM


_SEP = re.compile('/|\\\\') if IS_WINDOWS_PLATFORM else re.compile('/')


def tar(path, exclude=None, dockerfile=None, fileobj=None, gzip=False):
    root = os.path.abspath(path)
    exclude = exclude or []
    dockerfile = dockerfile or (None, None)
    extra_files = []
    if dockerfile[1] is not None:
        dockerignore_contents = '\n'.join(
            (exclude or ['.dockerignore']) + [dockerfile[0]]
        )
        extra_files = [
            ('.dockerignore', dockerignore_contents),
            dockerfile,
        ]
    return create_archive(
        files=sorted(exclude_paths(root, exclude, dockerfile=dockerfile[0])),
        root=root, fileobj=fileobj, gzip=gzip, extra_files=extra_files
    )


def exclude_paths(root, patterns, dockerfile=None):
    """
    Given a root directory path and a list of .dockerignore patterns, return
    an iterator of all paths (both regular files and directories) in the root
    directory that do *not* match any of the patterns.

    All paths returned are relative to the root.
    """

    if dockerfile is None:
        dockerfile = 'Dockerfile'

    patterns.append('!' + dockerfile)
    pm = PatternMatcher(patterns)
    return set(pm.walk(root))


def build_file_list(root):
    files = []
    for dirname, dirnames, fnames in os.walk(root):
        for filename in fnames + dirnames:
            longpath = os.path.join(dirname, filename)
            files.append(
                longpath.replace(root, '', 1).lstrip('/')
            )

    return files


def create_archive(root, files=None, fileobj=None, gzip=False,
                   extra_files=None):
    extra_files = extra_files or []
    if not fileobj:
        fileobj = tempfile.NamedTemporaryFile()
    t = tarfile.open(mode='w:gz' if gzip else 'w', fileobj=fileobj)
    if files is None:
        files = build_file_list(root)
    extra_names = {e[0] for e in extra_files}
    for path in files:
        if path in extra_names:
            # Extra files override context files with the same name
            continue
        full_path = os.path.join(root, path)

        i = t.gettarinfo(full_path, arcname=path)
        if i is None:
            # This happens when we encounter a socket file. We can safely
            # ignore it and proceed.
            continue

        # Workaround https://bugs.python.org/issue32713
        if i.mtime < 0 or i.mtime > 8**11 - 1:
            i.mtime = int(i.mtime)

        if IS_WINDOWS_PLATFORM:
            # Windows doesn't keep track of the execute bit, so we make files
            # and directories executable by default.
            i.mode = i.mode & 0o755 | 0o111

        if i.isfile():
            try:
                with open(full_path, 'rb') as f:
                    t.addfile(i, f)
            except OSError:
                raise OSError(
                    f'Can not read file in context: {full_path}'
                )
        else:
            # Directories, FIFOs, symlinks... don't need to be read.
            t.addfile(i, None)

    for name, contents in extra_files:
        info = tarfile.TarInfo(name)
        contents_encoded = contents.encode('utf-8')
        info.size = len(contents_encoded)
        t.addfile(info, io.BytesIO(contents_encoded))

    t.close()
    fileobj.seek(0)
    return fileobj


def mkbuildcontext(dockerfile):
    f = tempfile.NamedTemporaryFile()
    t = tarfile.open(mode='w', fileobj=f)
    if isinstance(dockerfile, io.StringIO):
        dfinfo = tarfile.TarInfo('Dockerfile')
        raise TypeError('Please use io.BytesIO to create in-memory '
                        'Dockerfiles with Python 3')
    elif isinstance(dockerfile, io.BytesIO):
        dfinfo = tarfile.TarInfo('Dockerfile')
        dfinfo.size = len(dockerfile.getvalue())
        dockerfile.seek(0)
    else:
        dfinfo = t.gettarinfo(fileobj=dockerfile, arcname='Dockerfile')
    t.addfile(dfinfo, dockerfile)
    t.close()
    f.seek(0)
    return f


def split_path(p):
    return [pt for pt in re.split(_SEP, p) if pt and pt != '.']


def normalize_slashes(p):
    if IS_WINDOWS_PLATFORM:
        return '/'.join(split_path(p))
    return p


def walk(root, patterns, default=True):
    pm = PatternMatcher(patterns)
    return pm.walk(root)


# Heavily based on
# https://github.com/moby/moby/blob/master/pkg/fileutils/fileutils.go
class PatternMatcher:
    def __init__(self, patterns):
        self.patterns = list(filter(
            lambda p: p.dirs, [Pattern(p) for p in patterns]
        ))
        self.patterns.append(Pattern('!.dockerignore'))

    def matches(self, filepath):
        matched = False
        parent_path = os.path.dirname(filepath)
        parent_path_dirs = split_path(parent_path)

        for pattern in self.patterns:
            negative = pattern.exclusion
            match = pattern.match(filepath)
            if not match and parent_path != '':
                if len(pattern.dirs) <= len(parent_path_dirs):
                    match = pattern.match(
                        os.path.sep.join(parent_path_dirs[:len(pattern.dirs)])
                    )

            if match:
                matched = not negative

        return matched

    def walk(self, root):
        def rec_walk(current_dir):
            for f in os.listdir(current_dir):
                fpath = os.path.join(
                    os.path.relpath(current_dir, root), f
                )
                if fpath.startswith('.' + os.path.sep):
                    fpath = fpath[2:]
                match = self.matches(fpath)
                if not match:
                    yield fpath

                cur = os.path.join(root, fpath)
                if not os.path.isdir(cur) or os.path.islink(cur):
                    continue

                if match:
                    # If we want to skip this file and it's a directory
                    # then we should first check to see if there's an
                    # excludes pattern (e.g. !dir/file) that starts with this
                    # dir. If so then we can't skip this dir.
                    skip = True

                    for pat in self.patterns:
                        if not pat.exclusion:
                            continue
                        if pat.cleaned_pattern.startswith(
                                normalize_slashes(fpath)):
                            skip = False
                            break
                    if skip:
                        continue
                yield from rec_walk(cur)

        return rec_walk(root)


class Pattern:
    def __init__(self, pattern_str):
        self.exclusion = False
        if pattern_str.startswith('!'):
            self.exclusion = True
            pattern_str = pattern_str[1:]

        self.dirs = self.normalize(pattern_str)
        self.cleaned_pattern = '/'.join(self.dirs)

    @classmethod
    def normalize(cls, p):

        # Remove trailing spaces
        p = p.strip()

        # Leading and trailing slashes are not relevant. Yes,
        # "foo.py/" must exclude the "foo.py" regular file. "."
        # components are not relevant either, even if the whole
        # pattern is only ".", as the Docker reference states: "For
        # historical reasons, the pattern . is ignored."
        # ".." component must be cleared with the potential previous
        # component, regardless of whether it exists: "A preprocessing
        # step [...]  eliminates . and .. elements using Go's
        # filepath.".
        i = 0
        split = split_path(p)
        while i < len(split):
            if split[i] == '..':
                del split[i]
                if i > 0:
                    del split[i - 1]
                    i -= 1
            else:
                i += 1
        return split

    def match(self, filepath):
        return fnmatch(normalize_slashes(filepath), self.cleaned_pattern)
