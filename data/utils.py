import ocnn
import numpy as np
from plyfile import PlyData

class ReadPly:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    self.has_normal = has_normal
    self.has_color = has_color
    self.has_label = has_label

  def __call__(self, filename: str):
    plydata = PlyData.read(filename)
    vtx = plydata['vertex']

    output = dict()
    points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
    output['points'] = points.astype(np.float32)
    if self.has_normal:
      normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
      output['normals'] = normal.astype(np.float32)
    if self.has_color:
      color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
      output['colors'] = color.astype(np.float32)
    if self.has_label:
      label = vtx['label']
      output['labels'] = label.astype(np.int32)
    return output

def random_point_dropout(pc, max_dropout_ratio=0.875):
    # 随机丢弃一部分点
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :] # 简单复制第一个点代替
    return pc

def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    # 随机缩放
    scales = np.random.uniform(scale_low, scale_high, 3)
    pc[:, 0:3] *= scales
    return pc

def shift_point_cloud(pc, shift_range=0.1):
    # 随机平移
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    pc[:, 0:3] += shifts
    return pc