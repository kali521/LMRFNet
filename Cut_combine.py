# -*-coding:utf-8-*-
"""
Created on 2022.5.1
programming language: python
@author:YeJianTingYu
"""
# '''
#     1. Pad the right and bottom of the original data block (arr1) to make its width and height divisible by patch size (L*L).
#     2. After processing patches through the network, only the central part (L * L) is kept. Reconstruct the original-sized data by arranging these patches in order.
# '''
import numpy as np

def cut(seismic_block, patch_size, stride_x, stride_y):
    """
    :param seismic_block: Input seismic data
    :param patch_size: Patch size
    :param stride_x: Horizontal stride size, should equal patch_size
    :param stride_y: Vertical stride size, should equal patch_size
    :return: Patches after padding (stored as list), number of patches in height direction, number in width direction
    """
    [seismic_h, seismic_w] = seismic_block.shape  # Get the height and width of the seismic data block
    # Pad the data to ensure complete patch coverage
    # Determine padded width dimension
    n1 = 1
    while (patch_size + (n1 - 1) * stride_x) <= seismic_w:
        # Determine how many steps can be taken with patch_size and stride_x on seismic_w
        n1 = n1 + 1
    # After loop, patch_size + (n1-1)*stride_x > seismic_w, meaning entire data can be covered with integer steps
    arr_w = patch_size + (n1 - 1) * stride_x
    # Determine padded height dimension
    n2 = 1
    while (patch_size + (n2 - 1) * stride_y) <= seismic_h:
        n2 = n2 + 1
    arr_h = patch_size + (n2 - 1) * stride_y
    # Pad the right and bottom of the seismic_block data block with zeros
    fill_arr = np.zeros((arr_h, arr_w), dtype=np.float32)
    fill_arr[0:seismic_h, 0:seismic_w] = seismic_block
    # After padding, the data to be sliced is the padded data
    # Calculate sliding step positions in arr_w direction
    # Python indexing starts at 0 (left-closed right-open). patch_size + (n-1)*stride_x = actual position + 1
    # The calculated n is one too large
    path_w = []  # Store x-direction sliding step positions
    x = np.arange(n1)  # Generate sequence [0~n1-1]
    x = x + 1  # Transform sequence to [1~n1]
    for i in x:
        s_x = patch_size + (i - 1) * stride_x  # Calculate each step position (actual position + 1)
        path_w.append(s_x)  # Add to list
    number_w = len(path_w)
    path_h = []
    y = np.arange(n2)
    y = y + 1
    for k in y:
        s_y = patch_size + (k - 1) * stride_y
        path_h.append(s_y)
    number_h = len(path_h)
    # Extract small patches from data using slice index positions
    cut_patches = []
    for index_x in path_h:  # path_h index is patch row
        for index_y in path_w:  # path_w index is patch column
            patch = fill_arr[index_x - patch_size: index_x, index_y - patch_size: index_y]
            cut_patches.append(patch)
    return cut_patches, number_h, number_w, arr_h, arr_w

def combine(patches, patch_size, number_h, number_w, block_h, block_w):
    """
    After slicing complete data with get_patches, reconstruct to original block size
    :param patches: Result from get_patches (list form)
    :param patch_size: Patch size
    :param number_h: Number of patches in height direction
    :param number_w: Number of patches in width direction
    :param block_h: Height of seismic data block
    :param block_w: Width of seismic data block
    :return: Reconstructed seismic data block
    """
    # Extract data from patch1 list, convert to 2D matrix, and concatenate in list element order
    # patch_size = int(patch_size)
    temp = np.zeros((int(patch_size), 1), dtype=np.float32)# Temporary concatenation matrix, to be deleted later
    print(temp.size)
    # Extract each element from patch1 and concatenate along column direction (axis=1)
    for i in range(len(patches)):
        temp = np.append(temp, patches[i], axis=1)

    # After deleting temp, temp1 dimension is patch_size * patch_size*number_h*number_w
    temp1 = np.delete(temp, 0, axis=1)  # Delete temp
    # Transform data to (patch_size*number_h) * (patch_size*number_w)
    test = np.zeros((1, int(patch_size*number_w)), dtype=np.float32)  # Temporary concatenation matrix, to be deleted later
    # Let temp1 switch to new line every patch_size*number_w columns
    for j in range(0, int(patch_size*number_h*number_w), int(patch_size*number_w)):
        test = np.append(test, temp1[:, j:j + int(patch_size*number_w)], axis=0)
    test1 = np.delete(test, 0, axis=0)  # Delete test
    block_data = test1[0:block_h, 0:block_w]
    return block_data
import numpy as np