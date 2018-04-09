import numpy as np
import os

class pickler:

    def __init__(self,cluster,prefix="data/"):
        self.cluster = cluster
        self._cluster_prefix = prefix
        # os.chdir(self._cluster_prefix)

    def load(self):
        if pickler.cluster_exists(self.cluster,self._cluster_prefix):
            data = np.load(self._cluster_prefix+self.get_filename())
        else:
            raise OSError
        return data

    def write(self,**data):
        np.savez(self._cluster_prefix+self.cluster,**data)

    @staticmethod
    def cluster_exists(cluster,prefix="data/"):
        filename = "%s%s.npz"%(prefix,cluster)
        if os.path.isfile(filename):
            return True
        return False

    def get_filename(self):
        # return "%s%s.npz"%(self._cluster_prefix, self.cluster)
        return "%s.npz"%(self.cluster)