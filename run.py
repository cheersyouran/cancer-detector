from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from utils import *
from datafile import *
import model
import scipy.ndimage as ndimage
import random

datagen = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
                rotation_range=90,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.4,
                width_shift_range=0.4,
                height_shift_range=0.4,
                )

model = model.generate_model()

descrimative_ind = {}

# random_eps个回合内不选择descrimative patch, 为了防止初始时的离谱选择
random_eps = 2

# 图片可视化，图片的id
check_num = 1

vis = {}
for e in range(15):
    for i in range(NUM_SAMPLE):
        img_id = train_sub.iloc[i, 0]
        label = train_sub.iloc[i, 1]
        print('')
        print('################ Img ####################', i, 'Label:', label)

        label_onehot = np_utils.to_categorical(label, 2)
        for img_path in img_path_li:
            if str(train_sub.iloc[i, 0]) in img_path:
                path = img_path
        img = load_img('Dataset_A/data/' + path, grayscale=True)
        x = img_to_array(img)
        x = add_zeros(x, PATCH_SIZE)
        org_patches = generate_patches(x, PATCH_SIZE)
        if e < random_eps:
            patches = org_patches.copy()
        else:
            patches = org_patches[descrimative_ind[i]]

        labels = np.array(label_onehot.tolist() * len(patches)).reshape((-1, 2))

        model.fit_generator(datagen.flow(patches, labels, batch_size=32, shuffle=False),
                    steps_per_epoch=len(patches) // 32, epochs=1, verbose=1)

        predictions = model.predict(org_patches)[:, label]

        if e >= random_eps:
            if label == 1:
                predictions[~descrimative_ind[i]] = random.uniform(0, 0.5)
            else:
                predictions[~descrimative_ind[i]] = random.uniform(0.5, 1)

        height = x.shape[0] // PATCH_SIZE
        width = x.shape[1] // PATCH_SIZE
        pred_reshape = predictions.reshape((height, width))

        # 为了防止descrimative的数量为零
        if np.where(gaussian_pred >= thresh)[0].size > org_patches.shape[0] * 0.05:
            descrimative_ind[i] = np.where(~(gaussian_pred <= thresh))[0]

        vis[i] = org_patches, descrimative_ind[i], x.shape


model.save('store/model.md')