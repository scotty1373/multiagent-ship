from imp import load_source
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), f'{name}.py')
    return load_source('', pathname)
