import argparse
import numpy as np
import os
import re
import cv2
from numpy import linalg as LA
from multiprocessing import Pool
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', dest='img_dir', default='')
parser.add_argument('--save_dir', dest='save_dir', default='')
parser.add_argument('--landmark_file', dest='landmark_file', default='./data/landmark.txt')
parser.add_argument('--standard_landmark_file', dest='standard_landmark_file', default='./standard_landmark_68pts.txt')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=224)
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=224)
parser.add_argument('--move_h', dest='move_h', type=float, default=0.25)
parser.add_argument('--move_w', dest='move_w', type=float, default=0.)
parser.add_argument('--save_format', dest='save_format', choices=['jpg', 'png'], default='jpg')
parser.add_argument('--n_worker', dest='n_worker', type=int, default=1)
parser.add_argument('--face_factor', dest='face_factor', type=float,
                    help='The factor of face area relative to the output image.', default=0.45)
parser.add_argument('--align_type', dest='align_type', choices=['affine', 'similarity'], default='similarity')
parser.add_argument('--order', dest='order', type=int, choices=[0, 1, 2, 3, 4, 5], help='The order of interpolation.',
                    default=3)
parser.add_argument('--mode', dest='mode', choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'], default='edge')
args = parser.parse_args()

with open(args.landmark_file) as f:
    line = f.readline()

n_landmark = len(re.split('[ ]+', line)[1:]) // 2
save_dir = os.path.join(args.save_dir, 'align_size(%d,%d)_move(%.3f,%.3f)_face_factor(%.3f)_%s' % (
args.crop_size_h, args.crop_size_w, args.move_h, args.move_w, args.face_factor, args.save_format))
img_names = np.genfromtxt(args.landmark_file, dtype=np.str, usecols=0)
landmarks = np.genfromtxt(args.landmark_file, dtype=np.float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1,
                                                                                                            n_landmark,
                                                                                                            2)
standard_landmark = np.genfromtxt(args.standard_landmark_file, dtype=np.float).reshape(n_landmark, 2)
standard_landmark[:, 0] += args.move_w
standard_landmark[:, 1] += args.move_h
celeba_landmark = np.genfromtxt("./celeba/Anno/list_landmarks_celeba.txt", dtype=np.float, usecols=range(1, 5 * 2 + 1),
                                skip_header=2).reshape(-1, 5, 2)


def work(i):
    try:
        src_landmarks = landmarks[i]
        src_celeba_landmarks = celeba_landmark[i]
        standard_landmarks = standard_landmark

        face_factor = 0.7
        crop_size_h = 224
        crop_size_w = 224
        # estimate transform matrix
        trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array(
            [crop_size_w // 2, crop_size_h // 2])

        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]

        # calcaute the scale of tform
        m1 = np.mat('0;0;1')
        m2 = np.mat('1;0;1')
        p1 = tform.dot(m1)
        p2 = tform.dot(m2)
        scale = LA.norm(p2 - p1)

        # change the translations part of the transformation matrix for downwarding vertically
        tform[1][2] = tform[1][2] + 20 * scale

        # get transformed landmarks
        tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]
        tformed_celeba_landmarks = \
            cv2.transform(np.expand_dims(src_celeba_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]
        name = os.path.splitext(img_names[i])[0] + '.' + args.save_format
        tformed_landmarks.shape = -1
        tformed_celeba_landmarks.shape = -1
        name_landmark_str = ('%s' + (' %.1f' * n_landmark * 2)) % ((name,) + tuple(tformed_landmarks))
        succeed = True
    except Exception as e:
        succeed = False
        print(e)
    if succeed:
        return name_landmark_str, ' '.join([name] + [str(int(x)) for x in tformed_celeba_landmarks])
    else:
        print('%s fails!' % img_names[i])


if __name__ == '__main__':

    pool = Pool(args.n_worker)

    values = list(tqdm.tqdm(pool.imap(work, range(len(img_names))), total=len(img_names)))
    name_landmark_strs = []
    str2 = []
    for x in values:
        name_landmark_strs.append(x[0])
        str2.append(x[1])

    pool.close()
    pool.join()
    landmarks_path = os.path.join(save_dir, 'tformed_landmark_68point.txt')
    with open(landmarks_path, 'w') as f:
        for name_landmark_str in name_landmark_strs:
            if name_landmark_str:
                f.write(name_landmark_str + '\n')

    landmarks_path = os.path.join(save_dir, 'tformed_landmark_5point.txt')
    with open(landmarks_path, 'w') as f:
        for name_landmark_str in str2:
            if name_landmark_str:
                f.write(name_landmark_str + '\n')
