import numpy as np

class helpers:

    def __init__(self):
        pass

    def augment(self, data, label, batch_size):
        aug_data = []
        aug_label = []
        for cls in [0,1]: # class 0,1 
            cls_idx = np.where(label == cls)
            tmp_data = data[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(batch_size / 2), 1, 3, 1000))

            for ri in range(int(batch_size / 2)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(batch_size / 2)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        
        return data, label
