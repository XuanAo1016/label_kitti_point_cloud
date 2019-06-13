import os
import sys
import math

import laspy
import numpy as np

import pymp

labels = {'Misc': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Car': 3,\
          'Van': 4, 'Truck': 5, 'Tram': 6, 'Person_sitting': 7}

class Object2d(object):
    ''' 2d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')

        # extract label, truncation, occlusion
        self.img_name = int(data[0]) # 'Car', 'Pedestrian', ...
        self.typeid = int(data[1]) # truncated pixel ratio [0..1]
        self.prob = float(data[2])
        self.box2d = np.array([int(data[3]),int(data[4]),int(data[5]),int(data[6])])

    def print_object(self):
        print('img_name, typeid, prob: %s, %d, %f' % \
            (self.img_name, self.typeid, self.prob))
        print('2d bbox (x0,y0,x1,y1): %d, %d, %d, %d' % \
            (self.box2d[0], self.box2d[1], self.box2d[2], self.box2d[3]))

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative
        self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

def load_velo_scan(velo_filename, dtype=np.float32):
    scan = np.fromfile(velo_filename, dtype=dtype).reshape((-1, 4))
    return scan

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    def __init__(self, root_dir):
        '''root_dir contains training and testing folders'''
        self.split_dir = root_dir
        self.num_samples = 7481

        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')

    def __len__(self):
        return self.num_samples

    def get_current_name(self, idx):
        return '%06d.las'%(idx)

    def get_calibration(self, idx):
        assert(idx<self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return Calibration(calib_filename)

    def get_lidar(self, idx, dtype=np.float64):
        assert(idx<self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        print(lidar_filename)
        return load_velo_scan(lidar_filename, dtype)

    def get_label_objects(self, idx):
        assert(idx<self.num_samples)
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return read_label(label_filename)

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2];

    return np.transpose(corners_3d)

def distance_2d(p):
    """calculate the distance from point to lidar"""
    return math.sqrt((p[0])**2 + (p[1])**2)

def get_lidar_points(pc_velo, objects, calib, distance):
    """"""
    boxes = []

    for obj in objects:
        if obj.type == 'DontCare':
            continue
        else:
            label = labels[obj.type]

        box3d_pts_3d = compute_box_3d(obj, calib.P)
        b = calib.project_rect_to_velo(box3d_pts_3d)

        #boxes for calculate if the points inside
        boxes.append([np.asarray([b[0, 0], b[0, 1], b[0, 2]]),\
                      np.asarray([b[1, 0], b[1, 1], b[1, 2]]),\
                      np.asarray([b[3, 0], b[3, 1], b[3, 2]]),\
                      np.asarray([b[4, 0], b[4, 1], b[4, 2]]), label])

    result_points = []

    xmax = xmin = pc_velo[0][0]
    ymax = ymin = pc_velo[0][1]
    zmax = zmin = pc_velo[0][2]

    for point in pc_velo:
        #if point within valid distance
        if distance_2d(point) < distance:
            inside = False
            #if point inside the box
            for box in boxes:
                u = np.cross((box[0] - box[2]), (box[0] - box[3]))
                v = np.cross((box[0] - box[1]), (box[0] - box[3]))
                w = np.cross((box[0] - box[1]), (box[0] - box[2]))
                # if inside the box
                if np.dot(u, box[0]) > np.dot(u, point[:3]) > np.dot(u, box[1]) and\
                   np.dot(v, box[0]) < np.dot(v, point[:3]) < np.dot(v, box[2]) and\
                   np.dot(w, box[0]) > np.dot(w, point[:3]) > np.dot(w, box[3]):
                    inside = True
                    result_points.append([point[0], point[1], point[2], box[4]])
                    break
            if not inside:
                result_points.append([point[0], point[1], point[2], 0])
            xmax = max(xmax, point[0])
            xmin = min(xmin, point[0])
            ymax = max(ymax, point[1])
            ymin = min(ymin, point[1])
            zmax = max(zmax, point[2])
            zmin = min(zmin, point[2])

    return result_points, [xmin, xmax, ymin, ymax, zmin, zmax]

def get_las_bb(pc_velo):
    """get bounding box of las file"""
    xmax = xmin = pc_velo[0][0]
    ymax = ymin = pc_velo[0][1]
    zmax = zmin = pc_velo[0][2]

    for point in pc_velo:
        xmax = max(xmax, point[0])
        xmin = min(xmin, point[0])
        ymax = max(ymax, point[1])
        ymin = min(ymin, point[1])
        zmax = max(zmax, point[2])
        zmin = min(zmin, point[2])

    return [xmin, xmax, ymin, ymax, zmin, zmax]

def write_las_bb(file_name, file, bound_box):
    """write las file"""
    l_header = laspy.header.Header()

    output_file = laspy.file.File(file_name, mode='w', header = l_header)

    output_file.header.min = [bound_box[0], bound_box[2], bound_box[4]]
    output_file.header.max = [bound_box[1], bound_box[3], bound_box[5]]
    output_file.header.offset = [0.0, 0.0, 0.0]
    output_file.header.scale = [0.001, 0.001, 0.001]

    result = np.array(file)
    output_file.X = (result[:,0] - output_file.header.offset[0]) / output_file.header.scale[0]
    output_file.Y = (result[:,1] - output_file.header.offset[1]) / output_file.header.scale[1]
    output_file.Z = (result[:,2] - output_file.header.offset[2]) / output_file.header.scale[2]
    output_file.classification = result[:,3].astype(int)
    output_file.close()

def main(argv):
    # get the information from input
    input_path  = sys.argv[1]
    output_path = sys.argv[2]
    distance    = float(sys.argv[3])
    tmp_argv    = sys.argv[4]
    num_thread  = int(sys.argv[5])

    if tmp_argv == 'False' or tmp_argv == 'false':
        with_label = False
    else:
        with_label = True

    print "input_path:  ", input_path
    print "output_path: ", output_path
    print "distance:    ", distance
    print "with_label:  ", with_label

    dataset = kitti_object(input_path)

    if with_label:
        with pymp.Parallel(num_thread) as p:
            for data_idx in p.range(363, len(dataset)):
                # Load data from dataset
                objects = dataset.get_label_objects(data_idx)
                pc_velo = dataset.get_lidar(data_idx, np.float32)[:,0:4]
                calib   = dataset.get_calibration(data_idx)

                result_points, bbox = get_lidar_points(pc_velo, objects, calib, distance)

                write_las_bb(output_path + dataset.get_current_name(data_idx), result_points, bbox)
    else:
        with pymp.Parallel(num_thread) as p:
            for data_idx in p.range(len(dataset)):
                pc_velo = dataset.get_lidar(data_idx, np.float32)[:,0:4]
                bbox = get_las_bb(pc_velo)
                write_las_bb(output_path + dataset.get_current_name(data_idx), pc_velo, bbox)

if __name__ == '__main__':
    main(sys.argv)