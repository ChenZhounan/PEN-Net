#!/usr/bin/python
# encoding: utf-8

import random
import os
import sys
import glob
from collections import Counter

import cv2
import pickle
from tqdm import tqdm
import lmdb
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import ImageFile, ImageDraw, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import cfg
from datasets import writeCache, draw_coors, coords_render, corrds2dxdys, resizeKeepRatio, draw_coors_add, corrds2dxdys_abso


def read_pot(f_path):
    fp = open(f_path, 'rb')
    samples = []
    all_bytes = fp.read()
    i = 0
    while i < len(all_bytes):
        # read head
        sample_size = int.from_bytes(
            all_bytes[i:i + 2], sys.byteorder,
            signed=False)  # bytes size of current character
        i += 2
        tag_code = int.from_bytes(
            all_bytes[i:i + 4][::-1][-2:], sys.byteorder,
            signed=False)  # Dword (int) type, GB2132 or GBK
        tag_char = all_bytes[i:i + 4][::-1][-2:-1].decode(
            'gbk') if tag_code < 256 else all_bytes[i:i +
                                                      4][::-1][-2:].decode('gbk')
        i += 4
        stroke_number = int.from_bytes(all_bytes[i:i + 2],
                                       sys.byteorder,
                                       signed=False)  # unsigned short type
        i += 2
        #  read stroke coordinate
        coordinates_size = sample_size - 8
        coordinates = []
        stroke = []
        for _ in range(coordinates_size):
            x = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=True)
            i += 2
            y = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=True)
            i += 2
            if (x, y) == (-1, 0):
                coordinates.append(stroke)
                stroke = []
            elif (x, y) == (-1, -1):
                break
            else:
                stroke.extend([x, y])
        assert len(
            coordinates
        ) == stroke_number, "stroke length should be equal to stroke_number"
        samples.append({
            'tag': tag_code,
            'tag_char': tag_char,
            'stroke_number': stroke_number,
            'coordinates': coordinates
        })
    fp.close()
    return samples


def read_gnt(gnt_path):
    samples = []
    fp = open(gnt_path, 'rb')
    all_bytes = fp.read()
    i = 0
    while i < len(all_bytes):
        sample_size = int.from_bytes(
            all_bytes[i:i + 4], sys.byteorder,
            signed=False)  # bytes size of current character
        i += 4
        tag_code = int.from_bytes(
            all_bytes[i:i + 2][-2:], sys.byteorder,
            signed=False)  # Dword (int) type, GB2132 or GBK
        tag_char = all_bytes[i:i + 2][::-1][-2:-1].decode(
            'gbk') if tag_code < 256 else all_bytes[i:i + 2].decode('gbk')
        # print(tag_char)
        i += 2
        width = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=False)
        i += 2
        height = int.from_bytes(all_bytes[i:i + 2],
                                sys.byteorder,
                                signed=False)
        i += 2
        bitmap = np.frombuffer(all_bytes[i:i + width * height], dtype=np.uint8)
        bitmap = bitmap.reshape(height, width)
        # cv2.imshow('bitmap', bitmap)
        # cv2.waitKey(0)
        i += width * height
        samples.append({
            'sample_size': sample_size,
            'tag_code': tag_code,
            'tag_char': tag_char,
            'width': width,
            'height': height,
            'bitmap': bitmap
        })
    fp.close()
    return samples


def rm_list_repeat_items(input_list):
    repeat_dict = dict(Counter(input_list))
    return [key for key, value in repeat_dict.items() if value == 1]


def intersection_list(list_a, list_b):
    return list(set(list_a).intersection(set(list_b)))


def alignment_gnt_pot(input_gnt, input_pot):
    gnt_tag_list = [temp_sample['tag_code'] for temp_sample in input_gnt]
    pot_tag_list = [temp_sample['tag'] for temp_sample in input_pot]
    gnt_tag_list_rm = rm_list_repeat_items(gnt_tag_list)
    pot_tag_list_rm = rm_list_repeat_items(pot_tag_list)
    intersection_tag_list = intersection_list(gnt_tag_list_rm, pot_tag_list_rm)
    output_gnt = [
        input_gnt[gnt_tag_list.index(i_tag)] for i_tag in intersection_tag_list
    ]
    output_pot = [
        input_pot[pot_tag_list.index(i_tag)] for i_tag in intersection_tag_list
    ]
    return output_gnt, output_pot


class PotDataset(Dataset):
    def __init__(self, alphbet_txt=None, is_train=True):
        lmdb_path = 'data/data_lmdb/pot/train' if is_train else 'data/data_lmdb/pot/test'
        root = get_dataset_path('train1.1' if is_train else 'test1.1')
        self._index_set = None
        if alphbet_txt is not None:
            with open(alphbet_txt, 'r') as fr:
                self.alphabet = fr.readlines()[0]
        else:
            self.alphabet = ''
        if not os.path.exists(lmdb_path):
            print('reading pot and generating lmdb cache file...')
            os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
            env = lmdb.open(lmdb_path, map_size=1099511627776)
            pots = sorted(glob.glob(os.path.join(root, '*.pot')))
            assert len(pots) > 0
            cnt = 0
            cache = {}
            for pot in tqdm(pots):
                samples = read_pot(pot)
                for sample in samples:
                    data = {
                        'tag_char': sample['tag_char'],
                        'coordinates': sample['coordinates']
                    }
                    if len(self.alphabet) > 0:
                        if data['tag_char'] not in self.alphabet:
                            continue
                    data_byte = pickle.dumps(data)
                    data_id = str(cnt).encode('utf-8')
                    cache[data_id] = data_byte
                    if cnt % 1000 == 0:
                        writeCache(env, cache)
                        cache = {}
                        # print('Written %d' % (cnt))
                    cnt += 1
            cache['num_sample'.encode('utf-8')] = str(cnt).encode()
            writeCache(env, cache)
            print('save {} samples to {}'.format(cnt, lmdb_path))
        self.lmdb = lmdb.open(lmdb_path,
                              max_readers=8,
                              readonly=True,
                              lock=False,
                              readahead=False,
                              meminit=False)
        self.is_train = is_train
        self.img_h = cfg.TRAIN.IMG_H  # 64
        self.img_w = cfg.TRAIN.IMG_W
        self.resize = resizeKeepRatio((self.img_w, self.img_h))
        self.absolute = True

        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(
                txn.get('num_sample'.encode('utf-8')).decode())

    def set_subset(self, sub_index_set):
        '''
        输入dataset子集的下标列表，最大值小于self.num_sample
        '''
        assert isinstance(sub_index_set, list)
        assert max(sub_index_set) < self.num_sample
        self._index_set = sub_index_set
        print('set subset of %s, new len is %s' % (type(self), len(self)))

    def __len__(self):
        if self._index_set is None:
            return self.num_sample
        else:
            return len(self._index_set)

    def __getitem__(self, index):
        index = index % (len(self))
        if self._index_set is not None:
            index = self._index_set[index]
        """渲染轨迹"""
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            char_tag, coords = data['tag_char'], data['coordinates']
        # print("len of coords is ", coords[0])
        thickness = random.randint(1, 3) if self.is_train else 2
        img_pil, coords_rend, _ = coords_render(coords,
                                                width=self.img_w,
                                                height=self.img_h,
                                                thickness=thickness)
        if self.absolute:
            label = corrds2dxdys_abso(coords_rend)
        else:    
            label = corrds2dxdys(coords_rend)

        img_tensor = (np.expand_dims(np.array(img_pil), axis=0) - 127.) / 127.
        return {
            'img': torch.Tensor(img_tensor),
            'label': torch.Tensor(label),
            'img_np': np.array(img_pil),
            'char_tag': char_tag
        }

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        max_len = max([s['label'].shape[0] for s in batch_data])
        EOS = torch.zeros(bs, max_len, 5)
        EOS[:, :, -1] = 1
        output = {
            'img': torch.zeros((bs, 1, self.img_h, self.img_w)),
            'label': EOS,
        }
        for i in range(bs):
            s = batch_data[i]['label'].shape[0]
            output['label'][i, :s] = batch_data[i]['label']
            output['img'][i] = batch_data[i]['img']
        return output
