from scipy.fftpack import dct, idct
from PIL import Image
import numpy as np
import random

def print_matrix(A):
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in A]))
    print('\n')

def level(img):
    return [[j - 128 for j in i] for i in img]

def unlevel(img):
    return [[j + 128 for j in i] for i in img]

def level_blocks(img):
    return [[[j - 128 for j in i] for i in block] for block in img]

def unlevel_blocks(img):
    return [[[j + 128 for j in i] for i in block] for block in img]

def img_to_blocks(img):
    blocks = []
    H = len(img)
    L = len(img[0])
    l = int(np.ceil(L / 8))
    h = int(np.ceil(H / 8))
    if l != L / 8 or h != H / 8:
        print("IMAGE SIZE IS NOT A MULTIPLE OF 8!")
        return False
    for j in range(h):
        for i in range(l):
            block = [[img[8 * j + k][8 * i + l] for l in range(8)] for k in range(8)]
            blocks.append(block)
    return blocks


def img_from_blocks(blocks, bpr):
    LR = bpr * 8
    LC = int(len(blocks) / bpr * 8)
    img = [[blocks[int(np.floor(i / 8) + np.floor(j / 8) * bpr)][j % 8][i % 8] for i in range(LR)] for j in range(LC)]
    return img


def img_to_6x6_blocks(img):
    blocks = []
    H = len(img)
    L = len(img[0])
    l = int(np.floor(L / 6))
    h = int(np.floor(H / 6))
    if l * 6 != L or h * 6 != H:
        print("IMAGE SIZE IS NOT A MULTIPLE OF 6!")
        return False
    padded_img = [[0 for _ in range(L + 2)] for _ in range(H + 2)]
    for row in range(H):
        for col in range(L):
            padded_img[row+1][col+1] = img[row][col]
    # fill in border with neighbouring cells to reduce artefacts
    for i in range(1,H+2):
        padded_img[i][0] = padded_img[i][1]
        padded_img[i][L+1] = padded_img[i][L]
    padded_img[0] = padded_img[1]
    padded_img[H+1] = padded_img[H]

    for j in range(h):
        for i in range(l):
            block = [[padded_img[6 * j + k][6 * i + m] for m in range(8)] for k in range(8)]
            blocks.append(block)
    
    print(blocks)
    print("blocs")
    exit(0)

    return blocks


def img_from_6x6_blocks(blocks, bpr):
    LR = bpr * 6
    LC = int(len(blocks) / bpr * 6)
    print("LR",LR)
    print("LC",LC)
    print("len blocks: ", len(blocks))
    img = [[blocks[int(np.floor(i / 6) + np.floor(j / 6) * bpr)][1 + j % 6][1 + i % 6] for i in range(LR)] for j in
           range(LC)]
    return img


def get_dct_table(N):
    table = [[1 / np.sqrt(N) for j in range(N)]]
    for i in range(1, N):
        table.append([np.sqrt(2 / N) * np.cos((2 * k + 1) * i * np.pi / (2 * N)) for k in range(N)])
    return table


def dct_blocks(img):
    result = []
    DCT = get_dct_table(8)  # len(img))
    for block in img:
        result.append(np.matmul(np.matmul(DCT, block), np.transpose(DCT)))
    return result

def dct_blocks2(img):
    DCT = get_dct_table(8)  # len(img))
    return np.matmul(np.matmul(DCT, img), np.transpose(DCT))

def idct_blocks(img):
    result = []
    iDCT = np.transpose(get_dct_table(8))  # len(img)))
    for block in img:
        result.append(np.matmul(np.matmul(iDCT, block), np.transpose(iDCT)))
    #result = [[[int(np.round(block[i][j])) for j in range(8)] for i in range(8)] for block in result]
    return result


def flatten(matrix):
    return [i for i in matrix]


def quantize(img):
    Q50 = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
           [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
           [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
    result = []
    for block in img:
        result.append([[int(np.round(block[i][j] / Q50[i][j])) for j in range(8)] for i in range(8)])
    return result

def quantize2(img):
    Q50 = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
           [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
           [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
    return [[img[i][j] / Q50[i][j] for j in range(8)] for i in range(8)]


def dequantize(img):
    Q50 = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
           [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
           [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
    result = []
    for block in img:
        result.append([[int(block[i][j] * Q50[i][j]) for j in range(8)] for i in range(8)])
        # np.floor(block[i][j] * Q50[i][j])) for i in range(8)] for j in range(8)])
    return result


def zigzag(block):
    res = []
    i = 0
    j = 0
    while len(res) != 64:
        res.append(block[i][j])
        if (i + j) % 2 == 1:
            if i == len(block) - 1:
                j += 1
            elif j == 0:
                i += 1
            else:
                i += 1
                j -= 1
        else:
            if j == len(block) - 1:
                i += 1
            elif i == 0:
                j += 1
            else:
                i -= 1
                j += 1
    return res

def zigzag2(l):
    i = 0
    j = 0
    res = []
    while len(res) != 64:
        res.append(i*8 + j)
        if (i + j) % 2 == 1:
            if i == l - 1:
                j += 1
            elif j == 0:
                i += 1
            else:
                i += 1
                j -= 1
        else:
            if j == l - 1:
                i += 1
            elif i == 0:
                j += 1
            else:
                i -= 1
                j += 1
    return res


def unzigzag(arr, L):
    res = [[0 for _ in range(L)] for _ in range(L)]
    i = 0
    j = 0
    while (i != L - 1 or j != L - 1) and len(arr) >= 1:
        res[i][j] = arr.pop(0)
        if (i + j) % 2 == 1:
            if i == L - 1:
                j += 1
            elif j == 0:
                i += 1
            else:
                i += 1
                j -= 1
        else:
            if j == L - 1:
                i += 1
            elif i == 0:
                j += 1
            else:
                i -= 1
                j += 1
    if len(arr) != 0:
        res[L - 1][L - 1] = arr.pop(0)
    return res


def compress(array, nb_elem):
    compressed_array = []
    for block in array:
        #print_matrix(block)
        block_arr = zigzag(block)
        #print(block_arr)
        temp = [64 - nb_elem, block_arr[0:nb_elem]]
        compressed_array.append(temp)
    return compressed_array

def decompress(array):
    result = []
    for block in array:
        temp = unzigzag(block[1], 8)
        result.append(temp)
    print("size of array", len(array))
    return result

def compress_image(image, nb_elem, blocks):
    return compress(quantize(dct_blocks(img_to_blocks(level(image))[0:blocks])), nb_elem)

def expand_image(image, bpr):
    return unlevel(img_from_blocks(idct_blocks(dequantize(decompress(image))), bpr))

def compress_image_6x6_client(image, nb_elem):
    return compress(quantize(dct_blocks(img_to_6x6_blocks(level(image)))), nb_elem)

def expand_image_6x6_server(image):
    return unlevel_blocks(idct_blocks(dequantize(decompress(image))))

def compress_image_6x6_server(image, nb_elem):
    return compress(quantize(dct_blocks(level_blocks(image))), nb_elem)

def expand_image_6x6_client(image, bpr):
    return unlevel(img_from_6x6_blocks(idct_blocks(dequantize(decompress(image))), bpr))

def conv(matrix, filter):
    X = len(matrix)
    Y = len(matrix[0])
    result = [[0 for _ in range(Y)] for _ in range(X)]
    for x in range(X):
        for y in range(Y):
            for i in range(3):
                for j in range(3):
                    if (x == 0 and i == 0) or (x == X - 1 and i == 2) or (y == 0 and j == 0) or (y == Y - 1 and j == 2):
                        pass
                    else:
                        result[x][y] += matrix[x - 1 + i][y - 1 + j] * filter[i][j]
    return result

def f(blocks,m,num_blocks):
    new_blocks = np.empty((m, m), dtype=object)
    for i in range(0,m):
        for j in range(0,m):
            aux = []
            for k in range(0,num_blocks):
                aux.append(blocks[k][i][j])
            new_blocks[i][j] = aux
    return new_blocks

def inverse_f(new_blocks, m, num_blocks):
    original_matrices = []
    matrix_size = m * m

    for k in range(num_blocks):
        matrix = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                index = i * m + j
                matrix[i][j] = new_blocks[index][k]
        original_matrices.append(matrix)

    return original_matrices

def permute_array(arr, positions):
    if len(arr) != len(positions):
        raise ValueError("Array and positions must have the same length")
    permuted_array = np.zeros_like(arr)
    for i, pos in enumerate(positions):
        permuted_array[i] = arr[pos]

    return permuted_array


def conv_parallel(blocks, filter):
    result_blocks = []
    for block in blocks:
        result = [[block[i][j] for j in range(8)] for i in range(8)]
        for x in range(1, 7):
            for y in range(1, 7):
                result[x][y] = 0
                for i in range(3):
                    for j in range(3):
                        result[x][y] += block[x - 1 + i][y - 1 + j] * filter[i][j]
        for x in range(1, 7):
            result[x][0] = 2 * result[x][1] - result[x][2]
            result[x][7] = 2 * result[x][6] - result[x][5]
        result[0] = [2*result[1][i] - result[2][i] for i in range(len(result[1]))]
        result[7] = [2*result[6][i] - result[5][i] for i in range(len(result[6]))]

        result_blocks.append(result)    
    return result_blocks


#######################################################
#############  > SET PARAMETERS HERE <  ###############
#######################################################
## IMAGE: Gradient
# img_matrix = [[155-2*i-5*j for i in range(18)] for j in range(18)]
# boat
# img_orig = Image.open(r"boat_image.png").convert('L')
# img_matrix_512 = np.asarray(img_orig)
# img_matrix = []
# for i in range(510):
#         img_matrix.append(img_matrix_512[i][:510])


# # Image.fromarray(np.asarray(img_matrix)).show()
# F = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]  # Filter
# cutoff = 30  # elements kept after quantization
# BpR = int(len(img_matrix[0]) / 6)  # number of blocks per row
# print("Blocks per row: ", BpR)

# blocks = img_to_6x6_blocks(level(img_matrix))

# for b in blocks: print_matrix(b)

# new_blocks = f(blocks,8,4)

# print(zigzag(new_blocks))

images = ["Sun.bmp",   "peppers_gray.bmp",
"boat.bmp",            "pirate.bmp",
"cameraman.bmp",       "livingroom.bmp",    "pirate1024.bmp",
"cameraman256.bmp",    "mandril.bmp",       "pirate256.bmp",
"cardinal1024.bmp",    "mandril256.bmp",    "sunset1024.bmp",
"cardinal2048.bmp",    "monument.bmp",      "sunset2048.bmp",
"office.bmp",
"lake.bmp"]

for image in images:
    img_orig = Image.open(r""+image).convert('L')
    img_matrix = np.asarray(img_orig)
    np.savetxt("csvs/"+image+".csv", img_matrix, delimiter=",",fmt='%i')

# blocks = quantize(dct_blocks(img_to_6x6_blocks(level(img_matrix))))
# for b in blocks:
#     print_matrix(b)

# for b in dct_blocks2(f(blocks,8,4)):
#     print_matrix(b)

# for b in quantize2(dct_blocks2(blocks)):
#     print_matrix(b)

# zigzag_array = zigzag2(8)
# print("zigzag list: ", zigzag_array)

# new_blocks = f(blocks,8,4)
# print("new blocks: ", new_blocks)
# new_blocks_permuted = permute_array(new_blocks,zigzag_array)
# print("new blocks permuted", new_blocks_permuted)

# compression = compress(quantize(dct_blocks(img_to_6x6_blocks(level(img_matrix)))),cutoff)


# new_blocks = f(blocks,8,4)
# for b in new_blocks:
#     print(b)
    
# blocks2 = inverse_f(new_blocks, 8, 4)


# Example usage:
# original_blocks = f(some_input_matrices, m, num_blocks)
# inverse_result = inverse_f(original_blocks, m, num_blocks)


#print(new_blocks)

# comp_img = compress_image_6x6_client(img_matrix, cutoff)
# img_server = expand_image_6x6_server(comp_img.copy())
# conv_img = conv_parallel(img_server,F)
# conv_comp_img = compress_image_6x6_server(img_server, cutoff)
# print("conv comp img length", len(conv_comp_img))
# conv_img_client = expand_image_6x6_client(conv_comp_img, BpR)

# img1 = compress_image_6x6_client(img_matrix,cutoff)
# img2 = expand_image_6x6_server(img1)
# img25 = conv_parallel(img2,F)
# img3 = compress_image_6x6_server(img25,cutoff)
# img4 = expand_image_6x6_client(img3,BpR)

# img23 = img_from_6x6_blocks(img2,BpR)
# img29 = img_from_6x6_blocks(img25,BpR)

#Image.fromarray(np.asarray(img_matrix)).show()
#Image.fromarray(np.asarray(img2)).show()
#Image.fromarray(np.asarray(img23)).show()
#Image.fromarray(np.asarray(img29)).show()
#Image.fromarray(np.asarray(img4)).show()

#Image.fromarray(np.asarray(img_matrix)).show()
#Image.fromarray(np.asarray(img_from_6x6_blocks(img_server,BpR))).show()
#Image.fromarray(np.asarray(conv_img)).show()
#Image.fromarray(np.asarray(conv_img_client)).show()

# IMAGE: mountain80
# img_orig = Image.open(r"C:\Users\merte\PycharmProjects\Compression\mountain80.png").convert('L')
# img_matrix = np.asarray(img_orig)
"""
B = 80                     # number of blocks, this should be a multiple of len(row)/8
F = [[0,-1,0],[-1,5,-1],[0,-1,0]]   # Filter
cutoff = 14                             # elements kept after quantization



BpR = int(len(img_matrix[0]) / 8) #number of blocks per row
#######################################################
#################  > code to run <  ###################
#######################################################

comp_img = compress_image(img_matrix, cutoff, B)
img_server = expand_image(comp_img,BpR)
conv_img = conv(img_server,F)
conv_comp_img = compress_image(conv_img, cutoff, B)
conv_img_client = expand_image(conv_comp_img,BpR)


Image.fromarray(np.asarray(img_matrix)).show()
Image.fromarray(np.asarray(img_server)).show()
Image.fromarray(np.asarray(conv_img)).show()
Image.fromarray(np.asarray(conv_img_client)).show()
"""
