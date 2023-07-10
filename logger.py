import os
import shutil


class Logger:
    def __init__(self, pathname):
        paths = os.path.normpath(pathname).split(os.sep)
        if paths[0] == 'config':
            del paths[0]
        fullpath = os.path.join('logs', *paths)

        if os.path.exists(fullpath):
            ans = input('file exists, remove it (y/n)?')
            if 'y' not in ans:
                return
        else:
            os.makedirs(fullpath)

        shutil.copy2(pathname, os.path.join(fullpath, 'config.yaml'))
        self.fd = open(os.path.join(fullpath, 'logger'), 'w')
        self.counter = 0

    def close(self):
        self.fd.close()

    def write(self, *args):
        # TODO: Refactoring
        print(*args, file=self.fd)
        if self.counter % 1 == 0:
            self.fd.flush()
        self.counter += 1
