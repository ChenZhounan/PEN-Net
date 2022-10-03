from .data_utils import writeCache, draw_coors, plot_dxdys, coords_render, corrds2dxdys, \
    resizeKeepRatio, dxdys_inter, plot_arrow, remove_overlap_ptrs, default_collate_fn_, ptrs_inter, draw_dot_coors, draw_coors_add, corrds2dxdys_abso
from .casia import PotDataset, PotGntDataset, PotEnglishDataset
from .iam import IamDataset
from .hindi import TamilDataset
from .japanese import JapaneseDataset


def get_dataset(dataset_names, is_train):
    def get_one_dataset(dataset_name: str, is_train):
        assert dataset_name in ['PotEnglishDataset', 'PotGntDataset', 'PotDataset', 'TamilDataset', 'JapaneseDataset']
        if dataset_name == 'PotEnglishDataset':
            dataset = PotEnglishDataset(alphbet_txt='dict_english.txt', is_train=is_train)
        elif dataset_name == 'PotGntDataset':
            dataset = PotGntDataset(alphbet_txt='dict_3755.txt', is_train=is_train)
        elif dataset_name == 'PotDataset':
            dataset = PotDataset(alphbet_txt='dict_3755.txt', is_train=is_train)
        elif dataset_name == 'TamilDataset':
            dataset = TamilDataset(is_train=is_train)
        elif dataset_name == 'JapaneseDataset':
            dataset = JapaneseDataset(is_train=is_train)
        else:
            raise NotImplementedError
        return dataset

    assert isinstance(dataset_names, str) or isinstance(dataset_names, list)
    if isinstance(dataset_names, str):
        dataset = get_one_dataset(dataset_names, is_train=is_train)
        return dataset
    elif isinstance(dataset_names, list):
        assert len(dataset_names) > 0
        first_name = dataset_names[0]
        dataset = get_one_dataset(first_name, is_train=is_train)
        for temp_name in dataset_names[1:]:
            dataset = dataset + get_one_dataset(temp_name, is_train=is_train)
        return dataset
    else:
        raise


if __name__ == '__main__':
    import numpy as np
    import cv2
    from torch.utils.data import DataLoader
    from tqdm import tqdm


    def static_len(dataset):
        import matplotlib.pyplot as plt
        len_ls = []
        print('samples: %d' % len(dataset))
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            temp_len = data['label'].shape[0]
            len_ls.append(temp_len - 1)
        len_np = np.array(len_ls)
        print('mean = %.4f, max = %.4f' % (len_np.mean(), len_np.max()))
        plt.hist(len_ls, bins=200)
        plt.title('points num')
        plt.show()


    def show_dataloader(tar_dataloader):
        for data in tar_dataloader:
            img_tensor, labels = data['img'], data['label']
            print(img_tensor.shape, labels.shape)
            for img, label in zip(img_tensor, labels):
                img = (img.squeeze().cpu().numpy() + 1) / 2
                canvas = draw_coors(label)
                cv2.imshow('input', img)
                cv2.imshow('canvas', canvas)
                cv2.waitKey(0)

    train_dataset = get_dataset(['JapaneseDataset', 'TamilDataset', 'PotDataset', 'PotEnglishDataset'], is_train=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=default_collate_fn_, batch_size=8, shuffle=True)

    show_dataloader(train_dataloader)
