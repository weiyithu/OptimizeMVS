import os
import imageio
import numpy as np
from plyfile import PlyData, PlyElement
import scipy.misc


def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)

# ------ read data ------
def read_from_list(fpath):
    res = []
    with open(fpath) as file:
        for line in file:
            res.append(line.strip())
    return res

def read_pc_from_pcd(fpath):
    pc = []
    with open(fpath) as pcd:
        for line in pcd.readlines()[11:len(pcd.readlines())-1]:
            strs = line.split(' ')
            x = float(strs[0])
            y = float(strs[1])
            z = float(strs[2].strip())
            pc.append(np.array([x, y, z]))
    return np.array(pc)

def read_pc_from_ply(fpath):
    plydata = PlyData.read(fpath)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    return np.stack([x, y, z], axis=1)

def imread(fname):
    im = imageio.imread(fname)/255.0
    if len(np.shape(im)) == 3:
        im = im[:, :, :3]
    return im
def imread_depth(fname):
    im = scipy.misc.imread(fname)
    return im

# ------ write data ------
def write_ply(points, filename, text=True):
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_pc_to_file(data, fpath):
    with open(fpath, 'w') as file:
        for i in range(np.shape(data)[0]):
            file.write('{0} {1} {2}\n'.format(data[i][0], data[i][1], data[i][2]))
def imsave(fname, array):
    scipy.misc.toimage(array, cmin=0.0, cmax=1.0).save(fname)

# ------ restore and save model ------
def restoreModelFromIt(path, opt, sess, saver, it):
    saver.restore(sess, './' + path + "/models_{0}/{1}_it{2}.ckpt".format(opt.group, opt.model, it))

def restoreModel(path, opt, sess, saver):
    saver.restore(sess, './' + path + "/{0}.ckpt".format(opt.load))

def saveModel(path, opt, sess, saver, it):
    saver.save(sess, path + "/models_{0}/{1}_it{2}.ckpt".format(opt.group, opt.model, it))


