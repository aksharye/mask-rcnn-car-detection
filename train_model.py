import pandas as pd
from mrcnn.utils import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import zeros
from numpy import asarray
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
import keras

# Model Parameters (STEPS = steps per epoch, EPOCHS = number of training iterations)
STEPS = 10
EPOCHS = 1

# Load dataset
total = pd.read_csv("labels_train.csv")
cars = pd.DataFrame(total[total['class_id']==1])

cars_training = cars[0:10000]
cars_testing = cars[70794:72794]

class CarsDataset(Dataset):
    def load_dataset(self, is_train=False):
        self.add_class("dataset", 1, "cars")
        if is_train:
            for i in range(0, len(cars_training.index)):
                print(i)
                imagefilename = cars_training.iloc[i]['frame']
                img_path = "images/" + imagefilename
                self.add_image('dataset', image_id=i, path=img_path)
        if not is_train:
            for i in range(0, len(cars_testing.index)):
                imagefilename = cars_testing.iloc[i]['frame']
                img_path = "images/" + imagefilename
                self.add_image('dataset', image_id=i+70794, path=img_path)
    def load_mask(self, image_id):
        mask = zeros([300, 480, 1], dtype='uint8')
        class_ids = list()
        xmin = cars.iloc[image_id]['xmin']
        ymin = cars.iloc[image_id]['ymin']
        xmax = cars.iloc[image_id]['xmax']
        ymax = cars.iloc[image_id]['ymax']
        row_s, row_e = ymin, ymax
        col_s, col_e = xmin, xmax
        mask[row_s:row_e, col_s:col_e] = 1
        class_ids.append(self.class_names.index('cars'))
        return mask, asarray(class_ids, dtype='int32')
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    
class TrainConfig(Config):
    NAME = "cars_cfg"
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = STEPS
    IMAGE_SIZE = [300, 480]

# Prepare training data
train_set = CarsDataset()
train_set.load_dataset(is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# Prepare testing data
test_set = CarsDataset()
test_set.load_dataset(is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

config = TrainConfig()

# Display example training image with actual bounding boxes
image = test_set.load_image(0)
print(image.shape)
mask, class_ids = test_set.load_mask(70794)
print(mask.shape)

plt.imshow(image)
plt.imshow(mask[:,:,0], cmap='gray', alpha = 0.5)
plt.show()

# Train model, outputs weights file
model = MaskRCNN(mode = "training", model_dir='./', config=config)
model.load_weights('models/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=EPOCHS, layers='heads')
