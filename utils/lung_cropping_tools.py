import copy

import cv2
import numpy as np
import torch
from skimage import measure

from utils.lung_cropping_model.deeplab import DeepLab


def refine_lung_region(imgsSegResult):
    imgs = np.where(imgsSegResult > 0.5, 1, 0)
    region = measure.label(imgs)
    props = measure.regionprops(region)
    numPix = []
    for ia in range(len(props)):
        numPix += [[props[ia].area, ia]]
    numPix = sorted(numPix, key=lambda x: x[0], reverse=True)
    firstBig, firstindex = numPix[0]
    if len(numPix) > 1:
        secondBig, secondindex = numPix[1]
    else:
        secondBig, secondindex = 0, firstindex

    if secondBig < firstBig / 4:
        minx, miny, minz, maxx, maxy, maxz = props[firstindex].bbox  # [minr, maxr),[minc, maxc)
        x, y, z = props[firstindex].coords[0]
        imgs[region != region[x, y, z]] = 0
        newrg = measure.label(imgs)
        newprops = measure.regionprops(newrg)
        bbox = [[minx, maxx], [miny, maxy], [minz, maxz]]
    else:
        minx1, miny1, minz1, maxx1, maxy1, maxz1 = props[firstindex].bbox  # [minr, maxr),[minc, maxc)
        minx2, miny2, minz2, maxx2, maxy2, maxz2 = props[secondindex].bbox  # [minr, maxr),[minc, maxc)

        x1, y1, z1 = props[firstindex].coords[0]
        x2, y2, z2 = props[secondindex].coords[0]

        imgs[region == region[x1, y1, z1]] = 65535
        imgs[region == region[x2, y2, z2]] = 65535
        imgs[imgs != 65535] = 0
        imgs[imgs == 65535] = 1
        bbox = [[min(minx1, minx2), max(maxx1, maxx2)], [min(miny1, miny2), max(maxy1, maxy2)],
                [min(minz1, minz2), max(maxz1, maxz2)]]
    return imgs, bbox


def get_crop_model():
    modelpath = 'utils/lung_cropping_model/model_lobe.pkl'
    tmp_state_dict = torch.load(modelpath, map_location='cpu')['weight']
    state_dict = {k.replace('module.', ''): v for k, v in tmp_state_dict.items()}
    crop_model = DeepLab()
    crop_model.load_state_dict(state_dict)
    crop_model.cuda()
    crop_model.eval()

    return crop_model


def learning_based_lung_cropping(CT, model=None, bs=36):
    if model is None:
        model = get_crop_model()
    data = CT
    imgs = []
    imgs1 = []
    for i in range(data.shape[2]):
        im = copy.deepcopy(data[..., i])
        wcenter = -500
        wwidth = 1500
        minvalue = (2 * wcenter - wwidth) / 2.0 + 0.5
        maxvalue = (2 * wcenter + wwidth) / 2.0 + 0.5

        dfactor = 255.0 / (maxvalue - minvalue)

        zo = np.ones(im.shape) * minvalue
        Two55 = np.ones(im.shape) * maxvalue
        im = np.where(im < minvalue, zo, im)
        im = np.where(im > maxvalue, Two55, im)
        im = ((im - minvalue) * dfactor).astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        imgs1.append(im)

        im = im / 255.
        im -= (0.485, 0.456, 0.406)
        im /= (0.229, 0.224, 0.225)
        im = torch.tensor(im).float().permute(2, 0, 1).unsqueeze(0)
        imgs.append(im)

    res = []
    m_batch = len(imgs) // bs
    if len(imgs) % bs != 0:
        m_batch += 1

    with torch.no_grad():
        for j in range(m_batch):
            x = copy.deepcopy(imgs[j * bs: (j + 1) * bs])

            x = torch.cat(x, dim=0).cuda()
            _, y = model(x)
            y = torch.softmax(y, dim=1)
            y = torch.argmax(y, dim=1).detach().cpu().numpy()
            # print("\033[1;32mBatch Pre Shape :{}\033[0m\n".format(y.shape))
            for one in range(y.shape[0]):
                res.append(y[one].astype(np.uint8))

    tmp = np.stack([im.astype(np.uint8) for im in res], axis=0)
    dilatedtmp = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i, im in enumerate(tmp):
        im = copy.deepcopy(im)
        dilated = cv2.dilate(im, kernel, 5)
        dilatedtmp.append(dilated)
    dilatedtmp = np.stack(dilatedtmp, axis=0)
    dilatedtmp, bbox = refine_lung_region(dilatedtmp)
    z1, z2 = bbox[0]
    y1, y2 = bbox[1]
    x1, x2 = bbox[2]

    return z1, z2, y1, y2, x1, x2


def conventional_lung_cropping(CT):
    _, _, d = CT.shape
    lung_seg = []
    for i in range(d):
        slice = CT[:, :, i].astype(np.uint8)
        _, binary_slice = cv2.threshold(slice, 0, 255, cv2.THRESH_BINARY)

        # floodfill
        im_floodfill = binary_slice.copy()
        h, w = binary_slice.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = binary_slice | im_floodfill_inv

        # erode
        kernel = np.ones((25, 25), np.uint8)
        im_out = cv2.erode(im_out, kernel, iterations=1)

        binary_slice = 255 - (binary_slice + (255 - im_out))
        _, binary_slice = cv2.threshold(binary_slice, 200, 255, cv2.THRESH_BINARY)

        # drop small connected areas
        labeled_img, num = measure.label(binary_slice, background=0, return_num=True)

        if num >= 3:
            dict = {}
            props = measure.regionprops(labeled_img)
            for idx in range(num):
                dict[idx] = props[idx].bbox_area
            sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
            max_idx, second_idx = sorted_dict[0][0] + 1, sorted_dict[1][0] + 1
            binary_slice = np.array((labeled_img == max_idx) | (labeled_img == second_idx)).astype(np.uint8) * 255

        lung_seg.append(binary_slice)

    # compute overlap along z axis, start from middle slice
    current_slice = lung_seg[d // 2]
    for index in range(d // 2, -1, -1):
        slice_to_check = lung_seg[index]
        labeled_img, num = measure.label(slice_to_check, background=0, return_num=True)
        blank_slice = np.zeros_like(slice_to_check)
        for i in range(1, num + 1):
            current_connected_area = np.array(labeled_img == i).astype(np.uint8)
            if np.sum(current_connected_area * current_slice) > 0:
                blank_slice += current_connected_area * 255
        lung_seg[index] = blank_slice
        current_slice = lung_seg[index]

    current_slice = lung_seg[d // 2]
    for index in range(d // 2, d):
        slice_to_check = lung_seg[index]
        labeled_img, num = measure.label(slice_to_check, background=0, return_num=True)
        blank_slice = np.zeros_like(slice_to_check)
        for i in range(1, num + 1):
            current_connected_area = np.array(labeled_img == i).astype(np.uint8)
            if np.sum(current_connected_area * current_slice) > 0:
                blank_slice += current_connected_area * 255
        lung_seg[index] = blank_slice
        current_slice = lung_seg[index]

    z1, z2, y1, y2, x1, x2 = 0, 0, 1000, -1, 1000, -1
    for index in range(d):
        if np.sum(lung_seg[index]) != 0 and np.sum(lung_seg[index - 1]) == 0:
            z1 = index
        if np.sum(lung_seg[index]) == 0 and np.sum(lung_seg[index - 1]) != 0:
            z2 = index

        labeled_img, num = measure.label(lung_seg[index], background=0, return_num=True)
        props = measure.regionprops(labeled_img)
        for prop in props:
            min_r, min_c, max_r, max_c = prop.bbox
            y1 = min_r if min_r < y1 else y1
            y2 = max_r if max_r > y2 else y2
            x1 = min_c if min_c < x1 else x1
            x2 = max_c if max_c > x2 else x2
    return z1, z2, y1, y2, x1, x2
