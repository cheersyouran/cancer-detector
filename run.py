from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from utils import *
from datafile import *
import model
import scipy.ndimage as ndimage
import random

datagen = ImageDataGenerator()

model = generate_model()

descrimative_ind = {}

# random_eps个回合内不选择descrimative patch, 为了防止初始时的离谱选择
random_eps = 1

# 图片可视化，图片的id
# check_num = 1

vis = {}
for i in range(10):
    for e in range(1, 5):
        img_id = train_sub.iloc[i, 0]
        label = train_sub.iloc[i, 1]
        print('')
        print('################ Img id:', img_id, '####################', i, '  Label:', label)

        label_onehot = np_utils.to_categorical(label, 2)
        for img_path in img_path_li:
            if str(train_sub.iloc[i, 0]) in img_path:
                path = img_path
        img = load_img('Dataset_A/data/' + path, grayscale=True)
        x = img_to_array(img)
        x = add_zeros(x, PATCH_SIZE)
        org_patches = generate_patches(x, PATCH_SIZE)
        if e <= random_eps:
            patches = org_patches.copy()
        else:
            patches = org_patches[descrimative_ind[i]]
        labels = np.array(label_onehot.tolist() * len(patches)).reshape((-1, 2))
        model.fit_generator(datagen.flow(patches, labels, batch_size=min(32, labels.size), shuffle=False),
                    steps_per_epoch=50, epochs=2, verbose=1)

        predictions = model.predict(org_patches)[:, label]

        height = x.shape[0] // PATCH_SIZE
        width = x.shape[1] // PATCH_SIZE

        if e > random_eps:
            predictions[~descrimative_ind[i]] = 0
            pred_reshape = predictions.reshape((height, width))

        # 高斯平滑
            gaussian_pred = ndimage.gaussian_filter(pred_reshape, sigma=SIGMA, order=0).flatten()
            thresh = np.percentile(gaussian_pred, THRESH)

        # 为了防止descrimative的数量为零
            if np.where(gaussian_pred >= thresh)[0].size > org_patches.shape[0] * 0.01:
                descrimative_ind[i] = np.where(gaussian_pred >= thresh)[0]

            vis[i] = org_patches, descrimative_ind[i], x.shape, img_id, label

        else:
         # 高斯平滑
            pred_reshape = predictions.reshape((height, width))
            gaussian_pred = ndimage.gaussian_filter(pred_reshape, sigma=SIGMA, order=0).flatten()

            thresh = np.percentile(gaussian_pred, 90)
         # 为了防止descrimative的数量为零
            if np.where(gaussian_pred >= thresh)[0].size > org_patches.shape[0] * 0.01:
                descrimative_ind[i] = np.where(predictions >= thresh)[0]
            vis[i] = org_patches, descrimative_ind[i], x.shape, img_id, label

        check_num = i
        keys = list(vis.keys())
        visualize_patches(vis[keys[check_num]][2], vis[keys[check_num]][0],
                          vis[keys[check_num]][1], vis[keys[check_num]][3], vis[keys[check_num]][4])