from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from utils import *
from datafile import *
from CONFIG import *
import model
import scipy.ndimage as ndimage
import random

datagen = ImageDataGenerator(
                horizontal_flip=True, 
                vertical_flip=True,
                )

model = generate_model()

pre_descrimative_ind, descrimative_ind = {}, {}

# In the first iteration, don't choose discriminative patches in order to avoid unsatisfied results.
random_eps = 1

# Visualization
# check_num = 1

vis = {}
for i in range(NUM_SAMPLE):
    # Five iterations at most
    for e in range(1, 6):
        img_id = train_sub.iloc[i, 0]
        label = train_sub.iloc[i, 1]
        print('')
        print('################ Img id:', img_id, '####################', i, '  Label:', label)

        label_onehot = np_utils.to_categorical(label, 2)
        for img_path in img_path_li:
            if str(train_sub.iloc[i, 0]) in img_path:
                path = img_path
        img = load_img(PATH + 'Dataset_A/data/' + path, grayscale=True)
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

        # Gaussian Smoothing and Thresholding
            gaussian_pred = ndimage.gaussian_filter(pred_reshape, sigma=SIGMA, order=0).flatten()
            thresh = np.percentile(gaussian_pred, THRESH)

        # Prevent the number of discriminative patches to be 0
            pre_descrimative_ind[i] = descrimative_ind[i]
            if np.where(gaussian_pred >= thresh)[0].size > org_patches.shape[0] * 0.01:
                descrimative_ind[i] = np.where(gaussian_pred >= thresh)[0]

            vis[i] = org_patches, descrimative_ind[i], x.shape, img_id, label
            if np.abs(len(descrimative_ind[i]) - len(pre_descrimative_ind[i])) <= 5:
                print('')
                print('++++++++++++++ Converges ... Go to next image +++++++++++++++')
                print('')
                break

        else:
         # Gaussian Smoothing and Thresholding
            pred_reshape = predictions.reshape((height, width))
            gaussian_pred = ndimage.gaussian_filter(pred_reshape, sigma=SIGMA, order=0).flatten()
            thresh = np.percentile(gaussian_pred, 90)
            
         # Prevent the number of discriminative patches to be 0
            if np.where(gaussian_pred >= thresh)[0].size > org_patches.shape[0] * 0.01:
                descrimative_ind[i] = np.where(predictions >= thresh)[0]
            vis[i] = org_patches, descrimative_ind[i], x.shape, img_id, label

        # Visualize for each iteration
        check_num = i
        keys = list(vis.keys())
        visualize_patches(vis[keys[check_num]][2], vis[keys[check_num]][0],
                          vis[keys[check_num]][1], vis[keys[check_num]][3], vis[keys[check_num]][4])



        