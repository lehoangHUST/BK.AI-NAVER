import matplotlib.pyplot as plt
import os

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def check_folder(folder1: str, folder2: str):
    dict_store = {folder1: os.listdir(folder1), folder2: os.listdir(folder2)}
    len_folder1 = len(dict_store[folder1])
    len_folder2 = len(dict_store[folder2])
    length = len_folder1 if len_folder1 <= len_folder2 else len_folder2
    idx = 0
    while idx < length:
        if dict_store[folder1][idx][0] != dict_store[folder2][idx][0]:
            if len_folder1 <= len_folder2:
                print("Not found in folder 1 and have found in folder2: ", dict_store[folder2][idx])
            else:
                print("Not found in folder 2 and have found in folderi: ", dict_store[folder1][idx])
            return False
        idx += 1
    return True