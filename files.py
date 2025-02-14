import fsspec_xrootd as xrdfs
import os

def get_rootfiles_local(path):
    return get_files_recursive_local(path, lambda f: f.endswith(".root"))

def get_files_recursive_local(rootpath, allowed=lambda f: True):
    result = []

    for root, dirs, files in os.walk(rootpath):
        for file in files:
            if allowed(file):
                result.append(os.path.join(root, file)) 
    
    return result


def get_rootfiles(hostid, path):
    fs = xrdfs.XRootDFileSystem(hostid = hostid)
    return get_files_recursive(fs, path,
                               lambda f : f.endswith(".root"), 
                               'root://%s/'%hostid)

def get_files_recursive(fs, rootpath, allowed = lambda f : True, prepend = ''):
    pathlist = fs.ls(rootpath)
    result = []
    for path in pathlist:
        if path['type'] == 'directory':
            result += get_files_recursive(fs, path['name'], allowed, prepend)
        elif path['type'] == 'file':
            if allowed(path['name']):
                result.append(prepend + path['name'])
        else:
            raise RuntimeError("Unexpected file type: {}".format(path['type']))
    return result
