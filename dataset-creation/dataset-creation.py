import numpy as np
import os
import cv2
from PIL import Image

def save_archive(archive, save_path):
    np.savez(save_path, images=archive['images'], offsets=archive['offsets'])


def generate_images(load_path, save_path, number_of_images=396288, max_offset=32, archive_size=9216, image_size=(128,128)):
    archive = {'images': [], 'offsets': []}
    archive_num = 0
    for image_i, f in enumerate(os.listdir(load_path)):
        if len(archive['images']) == archive_size:
            print('Saving archive number ' + str(archive_num))
            save_archive(archive, save_path + 'archive_' + str(archive_num))
            archive = {'images': [], 'offsets': []}
            archive_num += 1

        if archive_num * archive_size == number_of_images:
            print('Dataset generated')
            return
        
        image = cv2.imread(os.path.join(load_path, f), cv2.IMREAD_GRAYSCALE)

        if image.shape[0] < image_size[0] + 2 * max_offset or image.shape[1] < image_size[1] + 2 * max_offset:
            continue

        # image = image.reshape()
        # print(os.path.join(load_path, f), image.shape)
        center_x = np.random.randint(image_size[1] // 2 + max_offset, image.shape[1] - image_size[1] // 2 - max_offset - 1)
        center_y = np.random.randint(image_size[0] // 2 + max_offset, image.shape[0] - image_size[0] // 2 - max_offset - 1)
        offsets = np.random.randint(-max_offset, max_offset, (4, 2))
        points = np.array([
            [center_y - image_size[0] // 2,     center_x + image_size[1] // 2 - 1],
            [center_y - image_size[0] // 2,     center_x - image_size[1] // 2],
            [center_y + image_size[0] // 2 - 1, center_x - image_size[1] // 2],
            [center_y + image_size[0] // 2 - 1, center_x + image_size[1] // 2 - 1]
        ], dtype=np.float32)
        target_points = points + offsets
        target_points = target_points.astype(np.float32)
        # print(points)
        homography = cv2.getPerspectiveTransform(target_points[:,[1,0]], points[:,[1,0]])
        target_image = cv2.warpPerspective(image, homography, image.T.shape)
        
        points = points.astype(np.int32)
        target_points = target_points.astype(np.int32)
        # rgb_image = cv2.imread(os.path.join(load_path, f), cv2.IMREAD_COLOR)
        # cv2.polylines(rgb_image, [points[:,np.newaxis,[1,0]]], True, (0, 255, 0))
        # cv2.polylines(rgb_image, [target_points[:,np.newaxis,[1,0]]], True, (0, 0, 255))
        # print(points)
        # target_points = target_points.astype(np.int32)
        dataset_image1 = image[ points[1,0]:points[2,0]+1, points[1,1]:points[0,1]+1]
        dataset_image2 = target_image[ points[1,0]:points[2,0]+1, points[1,1]:points[0,1]+1]
        # print(dataset_image1.shape)
        # print(dataset_image2.shape)

        # cv2.imshow('image', rgb_image)
        # cv2.imshow('image1', dataset_image1)
        # cv2.imshow('image2', dataset_image2)
        # cv2.waitKey()

        dataset_image = np.stack((dataset_image1, dataset_image2), axis=2)
        archive['images'].append(dataset_image)
        archive['offsets'].append(np.reshape(offsets, -1))

    save_archive(archive, save_path)


if __name__ == '__main__':
    load_path = '/home/dominik/workspace/HomographyNet/homography-net/dataset/rawCOCO/train2017'
    save_path = '/home/dominik/workspace/HomographyNet/homography-net/dataset/train/'
    generate_images(load_path, save_path, 640, 32, 64)
