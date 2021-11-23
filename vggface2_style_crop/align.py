from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from functools import partial
from multiprocessing import Pool
import os
import re
import dlib
import cropper
import numpy as np
import tqdm

# ==============================================================================
# =                                      param                                 =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', dest='img_dir', default='./dataset')
parser.add_argument('--save_dir', dest='save_dir',
                    default="./cropped_dataset")
parser.add_argument('--landmark_file', dest='landmark_file', default='./data/landmark.txt')
parser.add_argument('--standard_landmark_file', dest='standard_landmark_file',
                    default='./dataset/standard_landmark_68pts.txt')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=224)
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=224)
parser.add_argument('--move_h', dest='move_h', type=float, default=0.25)
parser.add_argument('--move_w', dest='move_w', type=float, default=0.)
parser.add_argument('--save_format', dest='save_format', choices=['jpg', 'png'], default='jpg')
parser.add_argument('--n_worker', dest='n_worker', type=int, default=4)
parser.add_argument('--face_factor', dest='face_factor', type=float,
                    help='The facto r of face area relative to the output image.', default=0.8)
parser.add_argument('--align_type', dest='align_type', choices=['affine', 'similarity'], default='similarity')
parser.add_argument('--order', dest='order', type=int, choices=[0, 1, 2, 3, 4, 5], help='The order of interpolation.',
                    default=3)
parser.add_argument('--mode', dest='mode', choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'], default='edge')
args = parser.parse_args()
ignore_landmark = True
draw_landmark = False


def draw_face_landmark(name, landmarkToPaint, img_crop):
    # draw face's landmark
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4
    name_landmark_str = name
    for point in landmarkToPaint:
        point1 = int(point[0])
        point2 = int(point[1])
        point_tuple = (point1, point2)
        name_landmark_str += ' %.1f %.1f' % (point[0], point[1])
        image_draw = cv2.circle(img_crop, point_tuple, point_size, point_color, thickness)
    if landmarkToPaint.shape[0] == 5:
        landmarks_image_path = "./cropped_dataset/Landmark_painted_5_points"
    elif landmarkToPaint.shape[0] == 68:
        landmarks_image_path = "./cropped_dataset/Landmark_painted_68_points"
    landmarks_image_path = os.path.join(landmarks_image_path, name)
    imwrite(landmarks_image_path, image_draw)


# ==============================================================================
# =                                opencv first                                =
# ==============================================================================

_DEAFAULT_JPG_QUALITY = 95
import cv2

imread = cv2.imread
imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
align_crop = cropper.align_crop_opencv
print('Use OpenCV')

# ==============================================================================
# =                                     run                                    =
# ==============================================================================

# count landmarks
with open(args.landmark_file) as f:
    line = f.readline()
n_landmark = len(re.split('[ ]+', line)[1:]) // 2

# read data
img_names = os.listdir(args.img_dir)
landmarks = np.genfromtxt(args.landmark_file, dtype=np.float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1,
                                                                                                            n_landmark,
                                                                                                            2)

standard_landmark = np.genfromtxt(args.standard_landmark_file, dtype=np.float).reshape(n_landmark, 2)
standard_landmark[:, 0] += args.move_w
standard_landmark[:, 1] += args.move_h

# data dir
save_dir = os.path.join(args.save_dir, 'align_size(%d,%d)_move(%.3f,%.3f)_face_factor(%.3f)_%s' % (
    args.crop_size_h, args.crop_size_w, args.move_h, args.move_w, args.face_factor, args.save_format))
data_dir = os.path.join(save_dir, 'data')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

# load celeba 5 point landamrk
celeba_landmark = np.genfromtxt("./celeba/Anno/list_landmarks_celeba.txt", dtype=np.float,
                                usecols=range(1, 5 * 2 + 1), skip_header=2).reshape(-1, 5, 2)


def generate_landmark(img):
    # location of the model (path of the model).
    Model_PATH = "./models/shape_predictor_68_face_landmarks.dat"

    # now from the dlib we are extracting the method get_frontal_face_detector()
    # and assign that object result to frontalFaceDetector to detect face from the image with
    # the help of the 68_face_landmarks.dat model
    frontalFaceDetector = dlib.get_frontal_face_detector()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Now the dlip shape_predictor class will take model and with the help of that, it will show
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    # Now this line will try to detect all faces in an image either 1 or 2 or more faces
    allFaces = frontalFaceDetector(imageRGB, 0)

    # List to store landmarks of all detected faces
    allFacesLandmark = []

    # Below loop we will use to detect all faces one by one and apply landmarks on them
    for k in range(0, max(1, len(allFaces))):
        # dlib rectangle class will detecting face so that landmark can apply inside of that area
        faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                           int(allFaces[k].right()), int(allFaces[k].bottom()))

        # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)

        # Svaing the landmark one by one to the output folder
        for point in detectedLandmarks.parts():
            allFacesLandmark.append([point.x, point.y])
            if draw_landmark:
                img = cv2.circle(img, (point.x, point.y), 2, (0, 0, 255), 3)
                cv2.imshow("preview", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return np.array(allFacesLandmark)


def work(i):  # a single work
    for _ in range(3):  # try three times
        try:
            img = imread(os.path.join(args.img_dir, img_names[i]))

            img_landmark = generate_landmark(img) if ignore_landmark else landmarks[i]

            img_crop, tformed_landmarks, tformed_celeba_landmarks, tform = align_crop(img,
                                                                                      img_landmark,
                                                                                      standard_landmark,
                                                                                      celeba_landmark[i],
                                                                                      crop_size=(args.crop_size_h,
                                                                                                 args.crop_size_w),
                                                                                      face_factor=args.face_factor,
                                                                                      align_type=args.align_type,
                                                                                      order=args.order,
                                                                                      mode=args.mode)

            name = os.path.splitext(img_names[i])[0] + '.' + args.save_format
            path = os.path.join(data_dir, name)
            if not os.path.isdir(os.path.split(path)[0]):
                os.makedirs(os.path.split(path)[0])
            imwrite(path, img_crop)

            # draw the landmark before transformed
            if i <= 500:
                draw_face_landmark(name, tformed_celeba_landmarks, img_crop)
                draw_face_landmark(name, tformed_landmarks, img_crop)
            tformed_landmarks.shape = -1
            tformed_celeba_landmarks.shape = -1
            name_landmark_str = ('%s' + (' %.1f' * n_landmark * 2)) % ((name,) + tuple(tformed_landmarks))
            succeed = True
            break
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
