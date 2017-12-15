from CONFIG import *
import matplotlib.pyplot as plt

img_path_li = os.listdir(PATH + '/Dataset_A/data/')

def add_zeros(x, size=PATCH_SIZE, col=1):
    add_zeros_1 = np.zeros((x.shape[0], size, col))
    x_enlarge = np.concatenate((x, add_zeros_1), axis=1)
    add_zeros_0 = np.zeros((size, x_enlarge.shape[1], col))
    x_enlarge = np.concatenate((x_enlarge, add_zeros_0), axis=0)
    return x_enlarge

# Visualize patches
def visualize_patches(shape, org_patches, descrimative_ind, img_id, label, size=PATCH_SIZE):

    height, width, depth = shape
    col_size = width // size
    row_size = height // size

    dis_pathces = org_patches.copy()
    dis_pathces[:] = 0 # 黑色
    dis_pathces[descrimative_ind] = 255 # 白色

    org_patches = generate_image(col_size, row_size, org_patches)
    dis_pathces = generate_image(col_size, row_size, dis_pathces)

    fig = plt.figure()
    fig.add_subplot(121)
    plt.imshow(org_patches[:, :, 0], cmap='gray')
    fig.add_subplot(122)
    plt.imshow(dis_pathces[:, :, 0], cmap='gray')
    title = 'ID:' + str(img_id) + ', Label:' + str(label)
    fig.suptitle(title)
    plt.show()

# Use an image to generate patches
def generate_patches(x, size=PATCH_SIZE):
    height, width, depth = x.shape
    patches = []
    for i in range(height // size):
        for j in range(width // size):
            patch = x[i*size:(i+1)*size, j*size:(j+1)*size, :]
            patches.append(patch)
    patches = np.array(patches)
    return patches

# From patches generate images
def generate_image(col_size, row_size, org_patches):
    patches = None
    row_patches = None

    for i in range(row_size):
        for j in range(col_size):
            if j == 0:
                row_patches = org_patches[i * col_size + j]
            else:
                row_patches = np.hstack((row_patches, org_patches[i * col_size + j]))
        if i == 0:
            patches = row_patches
        else:
            patches = np.vstack((patches, row_patches))

    return patches
