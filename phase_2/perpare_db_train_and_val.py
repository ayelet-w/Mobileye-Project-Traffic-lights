import os
import json
import glob
import argparse
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import phase_1.run_attention as run_attention


def show_img_from_binFile(path):
    img_path = path + "\data.bin"
    label_path = path + "\label.bin"
    images = np.memmap(img_path, mode='r', dtype=np.uint8).reshape([-1] + [81, 81] + [3])
    labels = np.memmap(label_path, mode='r', dtype=np.uint8)
    for i in range(images.shape[0]):
        if labels[i]:
            fig = plt.figure()
            title = "traffic lights" if labels[i] else "not traffic lights"
            fig.suptitle(title)
            plt.imshow(images[i])
            print(i)
            plt.show(block=True)

def change_labels_val():
    index = [4, 22, 24, 30, 32, 33, 42, 48, 50, 58, 62, 108, 110, 114, 118, 140, 142, 164,166, 174, 177, 179, 210, 214,
     220, 221, 224, 234, 240, 242, 243, 272, 286, 289, 292, 340, 344, 349, 354, 355, 356, 360, 365, 368, 370,
     372, 374, 375, 378, 380, 381, 387, 388, 389,391, 394, 395, 397, 398, 399, 407, 413, 415, 417, 419, 420,
     422, 427, 428, 429, 430, 432, 435, 437, 445, 447, 449, 451, 455, 456, 457, 458, 459, 461, 462, 463, 464,
     505, 507, 511, 531, 532, 533, 537, 541, 563, 577, 578, 599, 669, 670, 673, 675, 676, 686, 690, 693, 694,
     696, 721, 722, 725, 726, 727, 740, 743, 745, 747, 749, 751, 753, 765, 767, 783, 790, 797, 799, 801, 811,
     815, 817, 819, 831, 869, 893, 950, 1041, 1045, 1046]
    label_path = 'C:/Users/User/PycharmProjects/mobileye_project/val_database' + "\label.bin"
    labels = np.memmap(label_path, mode='readwrite', dtype=np.uint8)
    for i in range(len(index)):
        labels[index[i]] = 0
    label_path = 'C:/Users/User/PycharmProjects/mobileye_project/val_database' + "\label_new.bin"
    with open(label_path, "w") as file:
        for l in labels:
            l.astype('uint8').tofile(file)


def change_labels_train():
    index = [11, 142, 188, 190, 192, 587, 760, 762, 763, 806, 807, 808, 820, 830, 836, 852, 874, 878, 888, 901, 939,
     941, 947, 959, 961, 963, 965, 967, 994, 1033, 1034, 1058, 1060, 1061, 1066, 1067, 1082, 1112, 1114, 1116,
     1118, 1120, 1122, 1123, 1130, 1132, 1133, 1144, 1148, 1154, 1156, 1158, 1162, 1168, 1170, 1172, 1176,1180,
     1181, 1184, 1190, 1192, 1194, 1196, 1200, 1202, 1203, 1206, 1213, 1218, 1222, 1224, 1242, 1243, 1244, 1254,
     1258, 1262, 1266, 1268, 1272, 1280, 1286, 1288, 1289, 1292, 1294, 1296, 1298, 1299, 1322, 1324, 1325, 1328,
     1330, 1331, 1332, 1333, 1340, 1346, 1360, 1376, 1388, 1433, 1435, 1490, 1493, 1495, 1513, 1551, 1559, 1563,
     1567, 1569, 1616, 1690, 1692, 1693, 1696, 1698, 1700, 1701, 1713, 1716, 1718, 1719, 1722, 1724, 1726, 1729,
     1731, 1732, 1738, 1740, 1746, 1777, 1780, 1787, 1816, 1818, 1890, 1892, 1894, 1896, 1900, 1902, 1978,
     2091, 2116, 2145, 2185, 2189, 2211, 2213, 2239, 2257, 2259, 2271, 2978, 3234, 3236, 3238, 3272, 3278, 3574,
     3575, 3694, 3704, 3706, 3708, 3979, 3986, 4000, 4154, 4156, 4158, 4162, 4163, 4166, 4210, 4212, 4214, 4216,
     4217, 4222, 4226, 4230, 4232, 4236, 4237, 4240, 4244, 4248, 4254, 4256, 4287, 4306, 4307, 4310, 4324, 4340,
     4342, 4352, 4354, 4404, 4418, 4510, 4511, 4512, 4518, 4520, 4522, 4524, 4526, 4527, 4532, 4534, 4536, 4550,
     4553, 4556,4558, 4560, 4568, 4571, 4584, 4586, 4596, 4604, 4610, 4621, 4630, 4634, 4647, 4654, 4658, 4660, 4662
     ,4664, 4666, 4668, 4670, 4672, 4675, 4678, 4681, 4884, 4688, 4690, 4691, 4684, 4696, 4700, 4709, 4712, 4714,
     4720, 4724, 4725, 4728, 4730, 4738, 4743, 4745, 4752, 4754, 4755, 4756, 4796, 4807, 4824, 4826, 4844, 4850, 4852,
     4880, 4882, 4898, 4900, 4904, 4906, 4944, 4968, 4969, 4989, 4995, 5027, 5029, 5030]
    label_path = 'C:/Users/User/PycharmProjects/mobileye_project/train_database' + "\label.bin"
    labels = np.memmap(label_path, mode='readwrite', dtype=np.uint8)
    for i in range(len(index)):
        labels[index[i]] = 0
    label_path = 'C:/Users/User/PycharmProjects/mobileye_project/train_database' + "\label_new.bin"
    with open(label_path, "w") as file:
        for l in labels:
            l.astype('uint8').tofile(file)

def save_to_binFile(crop_img, label, path):
    img_path = path + "\data.bin"
    label_path = path + "\label.bin"
    with open(img_path, "w") as file:
        for img in crop_img:
            img.astype('uint8').tofile(file)
    with open(label_path, "w") as file:
        for l in label:
            l.astype('uint8').tofile(file)
    #show_img_from_binFile(path)


def get_db_by_dir(original_images_path, gt_images_path, argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    if args.dir is None:
        args.dir = original_images_path
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        gtFine_image_json_path = gt_images_path + image[len(original_images_path):]
        gtFine_image_json_path = gtFine_image_json_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        json_data = json.load(open(gtFine_image_json_path))
        what = ['traffic light']
        objects = [o for o in json_data['objects'] if o['label'] in what]
        if not objects:
            flist.remove(image)

    result_crops_image = list()
    result_labels = list()

    for image in flist:
        red_x, red_y, green_x, green_y = run_attention.find_tfl_lights(image)
        x = red_x + green_x
        y = red_y + green_y
        gtFine_image_path = gt_images_path + image[len(original_images_path):]
        gtFine_image_path = gtFine_image_path.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        count_tfl = 0
        for i in range(len(x)):
            image_gtFine = np.array(Image.open(gtFine_image_path))
            if image_gtFine[y[i]][x[i]] == 19:
                crop_image = Image.open(image)
                crop_image = np.array(crop_image.crop((x[i] - 40, y[i] - 40, x[i] + 41, y[i] + 41)))
                plt.imshow(crop_image)
                result_crops_image.append(crop_image)
                result_labels.append(np.array([1]))
                count_tfl += 1
            else:
                if count_tfl > 0:
                    crop_image = Image.open(image)
                    crop_image = np.array(crop_image.crop((x[i] - 40, y[i] - 40, x[i] + 41, y[i] + 41)))
                    crop_gt = Image.open(gtFine_image_path)
                    crop_gt = np.array(crop_gt.crop((x[i] - 40, y[i] - 40, x[i] + 41, y[i] + 41)))
                    if 19 not in crop_gt.flatten():
                        result_crops_image.append(crop_image)
                        result_labels.append(np.array([0]))
                        count_tfl -= 1

    return result_crops_image, result_labels


def run_get_all_val_db():
    result_crops_image = list()
    result_labels = list()
    city_dir = ['frankfurt', 'lindau', 'munster']
    for i in range(len(city_dir)):
        path_leftImg8bit = 'C:/Users/User/PycharmProjects/mobileye_project/Image/leftImg8bit/val/'
        path_leftImg8bit += city_dir[i]
        path_gtFine = 'C:/Users/User/PycharmProjects/mobileye_project/Image/gtFine/val/'
        path_gtFine += city_dir[i]
        crops_image_temp, labels_temp = get_db_by_dir(path_leftImg8bit, path_gtFine)
        result_crops_image += crops_image_temp
        result_labels += labels_temp
    save_to_binFile(result_crops_image, result_labels, 'C:/Users/User/PycharmProjects/mobileye_project/val_database')


def run_get_all_train_db():
    result_crops_image = list()
    result_labels = list()
    city_dir = ['aachen', 'bochum', 'breman', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'humburg',
                'hanover', 'jena', 'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen',
                'ulm', 'weimar', 'zurich']
    for i in range(len(city_dir)):
        path_leftImg8bit = 'C:/Users/User/PycharmProjects/mobileye_project/Image/leftImg8bit/train/'
        path_leftImg8bit += city_dir[i]
        path_gtFine = 'C:/Users/User/PycharmProjects/mobileye_project/Image/gtFine/train/'
        path_gtFine += city_dir[i]
        crop_img, labels = get_db_by_dir(path_leftImg8bit, path_gtFine)
        result_crops_image += crop_img
        result_labels += labels
    save_to_binFile(result_crops_image, result_labels, 'C:/Users/User/PycharmProjects/mobileye_project/val_database')
