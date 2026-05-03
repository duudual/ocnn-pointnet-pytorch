import os
import numpy as np
import torch
from torch.utils.data import Dataset

import ocnn
from ocnn.octree import Octree, Points, merge_octrees
from .utils import ReadPly, random_point_dropout, random_scale_point_cloud, shift_point_cloud
CATEGORIES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]


def read_off(filepath):
    '''
        {points: N*3, normals: N*3}
    '''
    with open(filepath, 'r') as f:
        lines = f.readlines()
    vertices = []
    for line in lines:
        if line.startswith('OFF'):
            continue
        parts = line.strip().split()
        if len(parts) == 3:
            vertices.append(list(map(float, parts)))
    return np.array(vertices, dtype=np.float32)

def read_file(filename: str):
  filename = filename.replace('\\', '/')
  if filename.endswith('.ply'):
    read_ply = ReadPly(has_normal=True)
    return read_ply(filename)
  elif filename.endswith('.npz'):
    raw = np.load(filename)
    output = {'points': raw['points'], 'normals': raw['normals']}
    return output
  else:
    raise ValueError

def pc_normalize(points):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / (max_dist + 1e-8)
    return points


# ============================================================================
# PointNet++ Dataset - 固定 num_points 点数
# ============================================================================
class PointNetDataset(Dataset):

    def __init__(self, root_dir, split='train', num_points=1024):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        self.file_list = []
        for label, cat_name in enumerate(CATEGORIES):
            cat_dir = os.path.join(root_dir, cat_name, split)
            if os.path.exists(cat_dir):
                for fname in os.listdir(cat_dir):
                    if fname.endswith('.off'):
                        self.file_list.append((os.path.join(cat_dir, fname), label))
            else:
                print(f'warning: category {cat_name} does not exist in {root_dir}')

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        points = read_off(file_path)

        if len(points) < self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]
        elif len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]

        if self.split == 'train':
            points = random_point_dropout(points)
            points = random_scale_point_cloud(points)
            points = shift_point_cloud(points)
        points = pc_normalize(points)

        return {
            'points': torch.from_numpy(points).float(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def pointnet_collate_fn(batch):

    return {
        'points': torch.stack([item['points'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


def get_pointnet_dataloader(root_dir, batch_size=32, num_points=1024,
                          split='train', num_workers=4):
    dataset = PointNetDataset(root_dir, split=split, num_points=num_points)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=pointnet_collate_fn
    )
    return loader


# ============================================================================
# O-CNN Dataset - 点数可变 + merge_octrees
# ============================================================================
class OCNNDataset(Dataset):

    def __init__(self, root_dir, split: str, depth: int, full_depth: int, distort: bool, angle: tuple,
               interval: tuple, scale: float, jitter: float, flip: tuple,
               orient_normal: str = '', uniform: bool = False, **kwargs):
        super().__init__()

        # my :
        self.root_dir = root_dir
        self.split = split

        self.file_list = []
        for label, cat_name in enumerate(CATEGORIES):
            cat_dir = os.path.join(root_dir, cat_name, split)
            if os.path.exists(cat_dir):
                for fname in os.listdir(cat_dir):
                    if fname.endswith('.ply'):
                        self.file_list.append((os.path.join(cat_dir, fname), label))

        # ocnn :
        # for octree building
        self.depth = depth
        self.full_depth = full_depth

        # for data augmentation
        self.distort = distort
        self.angle = angle
        self.interval = interval
        self.scale = scale
        self.uniform = uniform
        self.jitter = jitter
        self.flip = flip

        # for other transformations
        self.orient_normal = orient_normal

    def preprocess(self, sample: dict, idx: int):
        r''' Transforms :attr:`sample` to :class:`Points` and performs some specific
        transformations, like normalization.
        '''

        xyz = torch.from_numpy(sample.pop('points'))
        normals = torch.from_numpy(sample.pop('normals'))
        sample['points'] = Points(xyz, normals)
        return sample
    
    def transform(self, sample: dict, idx: int):
        r''' Applies the general transformations provided by :obj:`ocnn`.
        '''

        # The augmentations including rotation, scaling, and jittering.
        points = sample['points']
        if self.distort:
            rng_angle, rng_scale, rng_jitter, rnd_flip = self.rnd_parameters()
            points.flip(rnd_flip)
            points.rotate(rng_angle)
            points.translate(rng_jitter)
            points.scale(rng_scale)

        if self.orient_normal:
            points.orient_normal(self.orient_normal)

        # !!! NOTE: Clip the point cloud to [-1, 1] before building the octree
        inbox_mask = points.clip(min=-1, max=1)
        sample.update({'points': points, 'inbox_mask': inbox_mask})
        return sample

    def points2octree(self, points: Points):
        r''' Converts the input :attr:`points` to an octree.
        '''

        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree
    
    def rnd_parameters(self):
        r''' Generates random parameters for data augmentation.
        '''

        rnd_angle = []
        for i in range(3):
            rot_num = self.angle[i] // self.interval[i]
            rnd = torch.randint(low=-rot_num, high=rot_num+1, size=(1,))
            rnd_angle.append(rnd * self.interval[i] * (3.14159265 / 180.0))
        rnd_angle = torch.cat(rnd_angle)

        rnd_scale = torch.rand(3) * (2 * self.scale) - self.scale + 1.0
        if self.uniform:
            rnd_scale[1] = rnd_scale[0]
            rnd_scale[2] = rnd_scale[0]

        rnd_flip = ''
        for i, c in enumerate('xyz'):
            if torch.rand([1]) < self.flip[i]:
                rnd_flip = rnd_flip + c

        rnd_jitter = torch.rand(3) * (2 * self.jitter) - self.jitter
        return rnd_angle, rnd_scale, rnd_jitter, rnd_flip
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        sample = read_file(file_path)

        # preprocess first (convert to Points), then transform, then build octree
        # This matches thsolver's Transform.__call__ method
        output = self.preprocess(sample, idx)
        output = self.transform(output, idx)
        output['label'] = label

        # Generate octree from points
        output['octree'] = self.points2octree(output['points'])

        return output

def get_ocnn_dataloader(root_dir, batch_size=32, split='train',
                       depth=5, full_depth=2, distort=None,
                       angle=(0, 0, 5), interval=(1, 1, 1), scale=0.25,
                       jitter=0.125, flip=(0, 0, 0), orient_normal='xyz',
                       uniform=False, num_workers=4):
    if distort is None:
        distort = (split == 'train')
    dataset = OCNNDataset(
        root_dir=root_dir, split=split,
        depth=depth, full_depth=full_depth,
        distort=distort, angle=angle, interval=interval,
        scale=scale, jitter=jitter, flip=flip,
        orient_normal=orient_normal, uniform=uniform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=ocnn.dataset.CollateBatch()
    )
    return loader