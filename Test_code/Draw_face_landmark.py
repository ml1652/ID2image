import cv2
import os

def draw_face_landmark(name, landmarkToPaint,image_directory):
    point_size = 1
    point_color = (0, 0, 255)
    thickness = 4
    _DEAFAULT_JPG_QUALITY = 95
    img_crop =cv2.imread(name)
    image_draw = img_crop

    for landmark in landmarkToPaint:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, point_color, thickness)
    landmarks_image_path = os.path.join(image_directory, os.path.basename(name))
    cv2.imwrite(landmarks_image_path, image_draw)