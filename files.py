import os

def get_files_recursive(rootpath, allowed=lambda f: f.endswith(".root"), prepend=''):
    result = []
    for root, dirs, files in os.walk(rootpath):
        for file in files:
            if allowed(file):
                result.append(os.path.join(prepend, root, file))
    return result

#rootpath = '/hildafs/projects/phy230010p/share/ECONAE/training/data'
#files = get_files_recursive(rootpath)
