import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mrcnn.utils import Dataset
from numpy import zeros
from numpy import asarray
from mrcnn.config import Config
from numpy import expand_dims
from mrcnn.model import MaskRCNN
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import keras
from os import listdir
from mrcnn.visualize import display_instances

# PARAMETERS
# Path to model weights
model_path = 'models/mask_rcnn_cars_cfg_0001.h5'
# Images for prediction
testImages = ['1479506176491553178.jpg']

# Load images
total = pd.read_csv("labels_train.csv")
cars = pd.DataFrame(total[total['class_id']==1])
images = []
images_dir = 'images/'

for filename in listdir(images_dir):
    images.insert(0, filename)

# Load dataset
class CarsDataset(Dataset):

    def load_dataset(self):

        self.add_class("dataset", 1, "cars")

        for i in range(0,len(testImages)):
            imagefilename = testImages[i]
            img_path = "images/" + imagefilename
            self.add_image('dataset', image_id=i, path=img_path, filename = imagefilename)

    def load_mask(self, image_id):
        imagefilename = self.image_name(image_id)

        boxes = list()
        for i in range(0, len(cars.index)):
            if cars.iloc[i]['frame'] == imagefilename:
                xmin = cars.iloc[i]['xmin']
                ymin = cars.iloc[i]['ymin']
                xmax = cars.iloc[i]['xmax']
                ymax = cars.iloc[i]['ymax']
                boundary = [xmin, ymin, xmax, ymax]
                boxes.append(boundary)

        mask = zeros([300, 480, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            mask[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('cars'))
        return mask, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def image_name(self, image_id):
        info = self.image_info[image_id]
        return info['filename']

class PredictionConfig(Config):
    NAME = "cars_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Plot bounding boxes
def plot(dataset, model, cfg, n_images):

    for j in range(n_images):
        image = dataset.load_image(j)
        mask, _ = dataset.load_mask(j)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]
        plt.imshow(image)
        plt.title("Predicted Bounding Boxes")
        ax = plt.gca()
        for box in yhat["rois"]:
           y1,x1,y2,x2 = box
           width, height = x2 - x1, y2 - y1
           rect = Rectangle((x1,y1), width, height, fill=False, color='red')
           ax.add_patch(rect)
        plt.show()

# Prepare test set
test_set = CarsDataset()
test_set.load_dataset()
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# Predict
config = PredictionConfig()
model = MaskRCNN(mode = "inference", model_dir='./', config=config)
model.load_weights(model_path, by_name = True)

# Display results
image_id = 0
image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(test_set, config, image_id, use_mini_mask = False)
results = model.detect([image], verbose = 1)
r = results[0]
display_instances(image, r['rois'], r['masks'], r['class_ids'], test_set.class_names, r['scores'], title="Predicted Masks")



