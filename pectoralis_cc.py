import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from tensorflow import convert_to_tensor
import seaborn as sns
from tensorflow import numpy_function, float32
from tensorflow import function as tf_function
import tensorflow.keras.backend as K
from os.path import join, isfile
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid


# Constants
MODELS_FOLDER = 'models/'
TESTSET_FOLDER = 'testset/'
CSV_PATH = 'Pectoralis (CC) .csv'

# Grid Search Parameters

THRESHOLDS_CLASS = [0.5]
THRESHOLDS_SEG = [0.6, 0.7, 0.8]
OPENING_KERNELS = [0, 5, 10] 
CLOSING_KERNELS = [0] 
IMAGE_SIZE_224 = (224, 224)
IMAGE_SIZE_256 = (256, 256)
PLOTS_FOLDER = 'plots/'
METRICS_CSV_PATH = 'metrics.csv'


def _reshape_to_square(img):
    """
    Pastes a black strip on the right side of the image in order
    to make it square.

    :param img: Image to be processed
    :type img: PIL object
    """
    long_dim = max(img.size[0], img.size[1])
    new_im = Image.new("L", (long_dim, long_dim))
    new_im.paste(img, (0, 0))
    return new_im


def _prepare_input_data(image, input_shape, flip, crop=None, channels=1, square=True):
    """
    Processes an image and returns a numpy array that if suitable as the model input.

    :param image: Image to be processed
    :type image: PIL object
    :param input_shape: Target resolution
    :type input_shape: list or tuple
    :param flip: Determines if image needs to be flipped
    :type flip: boolean
    :param crop: Crop of the image
    :type crop: list or tuple
    :param channels: Number of channels (e.g. 3 for RGB)
    :type channels: int
    """
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if square:
        image = _reshape_to_square(image)
    if crop:
        # applies to coordinates in squared image (!)
        # left, top, right, bottom
        scaled_crop = (crop[0] * image.size[0],
                        crop[1] * image.size[1],
                        crop[2] * image.size[0],
                        crop[3] * image.size[1])
        image = image.crop(scaled_crop)
    resized = image.resize(input_shape)
    pixel_array = img_to_array(resized)
    pixel_array /= 255.0
    final_array = np.broadcast_to(
        pixel_array, (1, pixel_array.shape[0], pixel_array.shape[1], channels))

    return convert_to_tensor(final_array)


def compare_image_sides(image_path: str) -> str:
    img = Image.open(image_path)
    img_array = np.array(img)
    mid_index = img_array.shape[1] // 2
    left_sum = np.sum(img_array[:, :mid_index])
    right_sum = np.sum(img_array[:, mid_index:])
    return "left" if left_sum > right_sum else "right"
    

def _load_one_model(folder, model_name, custom_object=None):
    """
    Loads one model to the memory.
    :param folder: Folder containing the model
    :type folder: str
    :param model_name: Name of the model
    :type model_name: str
    """
    try:
        h5_path = join(folder, '{}.h5'.format(model_name))
        if not isfile(h5_path):
            return None
        model = load_model(h5_path, custom_objects=custom_object)
        print(f"Model {model_name} loaded successfully.")
        return tf_function(model)
    except Exception as err:
        print(err)
    

def _morph_open(input_arr, k_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    result = cv2.erode(input_arr, kernel)
    result = cv2.dilate(result, kernel)
    return result


def _morph_close(input_arr, k_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    result = cv2.dilate(input_arr, kernel)
    result = cv2.erode(result, kernel)
    return result


def _predict_overlays(input_arr, model, **kwargs):
        threshold = kwargs.get('threshold')
        class_idx = kwargs.get('class_idx')
        opening_kernel = kwargs.get('opening_kernel')
        closing_kernel = kwargs.get('closing_kernel')
        median_filter_kernel = kwargs.get('median_filter_kernel')
        mask = kwargs.get('mask')

        output = model(input_arr)
        output = output.numpy()[0]
        shape = (output.shape[0], output.shape[1])

        if not threshold:
            output = np.argmax(output, axis=-1)
            if median_filter_kernel:
                output = cv2.medianBlur(output.astype(
                    np.uint8), median_filter_kernel)

        if class_idx:
            n_classes = 1
            iterator = [class_idx-1]
        else:
            n_classes = output.shape[-1] - 1
            iterator = range(n_classes)

        overlays = n_classes * [None]

        for i in iterator:
            cls_map = np.zeros(shape, dtype=np.uint8)

            if threshold:
                cls_map[output[:, :, i+1] >= threshold] = 255
            else:
                cls_map[output == i+1] = 255

            if mask is not None:
                mask_resized = cv2.resize(mask, dsize=cls_map.shape)
                cls_map[mask_resized > 0] = 0

            if opening_kernel:
                cls_map = _morph_open(cls_map, opening_kernel)

            if closing_kernel:
                cls_map = _morph_close(cls_map, closing_kernel)

            overlays[i] = cls_map

        return overlays 
    

def _predict_class(model, input_data, threshold=None, print_proba=False):
    """
    Performs and returns a prediction for a given model and input image.

    :param model: Model to be applied.
    :type model: KERAS model
    :param input_data: Input numpy array
    :type input_data: Numpy array
    :param threshold: If not None, the bad predictions of probability
                        lower than its value, become moderate.
    :type threshold: float
    :param print_proba: If True, the result is printed for debugging purposes.
    :type print_proba: boolean
    """
    if model:
        softmax_prediction = model(input_data)[0]

        if print_proba:
            print(softmax_prediction)
        
        predicted_class = K.argmax(softmax_prediction)
        proba = softmax_prediction[1]
        if proba > threshold:
            predicted_class = 'insufficient'
        else: 
            predicted_class = 'correct'
        return predicted_class
    else:
        return None


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return numpy_function(f, [y_true, y_pred], float32)


def iou_no_bcg(y_true, y_pred):
    y_true = y_true[:, :, :, 1:]
    y_pred = y_pred[:, :, :, 1:]

    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return numpy_function(f, [y_true, y_pred], float32)


def load_csv_as_dataframe(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    return df


def get_bounding_box(input_shape, contour, aspect_ratio, flip, box_margins=None):
    if not box_margins:
        box_margins = (0, 0)

    x, y, w, h = cv2.boundingRect(contour)  # bottom left

    x /= input_shape[1]
    w /= input_shape[1]
    y /= input_shape[0]
    h /= input_shape[0]

    if aspect_ratio:
        x *= aspect_ratio
        w *= aspect_ratio

    if flip:
        x = 1 - x
        x -= w

    # add margin
    x = max(0, x-(box_margins[0] / 2))
    y = max(0, y-(box_margins[1] / 2))
    w += box_margins[0]
    if x + w > 1:
        w = 1 - x
    h += box_margins[1]
    if y + h > 1:
        h = 1 - y

    return x, y, w, h


def find_contours(input_arr, get_boxes=True, **kwargs):
    biggest_only = kwargs.get('biggest_only')
    original_img_size = kwargs.get('original_img_size')
    flip = kwargs.get('flip')
    box_margins = kwargs.get('box_margins')

    contours, _ = cv2.findContours(input_arr,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    if biggest_only:
        contours = [max(contours, key=cv2.contourArea)]
        input_arr = np.zeros(input_arr.shape, np.uint8)
        cv2.drawContours(input_arr, contours, -1, 1, cv2.FILLED)

    n = len(contours)
    contours_processed = n * [None]
    boxes = []

    aspect_ratio = None
    if original_img_size:
        aspect_ratio = original_img_size[1] / original_img_size[0]

    for i in range(n):
        cnt_temp = []
        for point in contours[i]:
            point_temp = [0, 0]
            point_temp[0] = point[0][0] / input_arr.shape[1]
            point_temp[1] = point[0][1] / input_arr.shape[0]

            if aspect_ratio:
                point_temp[0] *= aspect_ratio
                if flip:
                    point_temp[0] = 1 - point_temp[0]

            cnt_temp.append(point_temp)

        if get_boxes:
            box = get_bounding_box(input_arr.shape, contours[i],
                                aspect_ratio, flip,
                                box_margins=box_margins)
            boxes.append(box)

        contours_processed[i] = cnt_temp

    result = {
        'contours': contours_processed
    }

    if get_boxes:
        result['boxes'] = boxes

    return result


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1


def is_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
      

def merge_boxes(boxes, contours):
    n = len(boxes)
    ds = DisjointSet(n)

    # Build disjoint sets based on overlapping boxes
    for i in range(n):
        for j in range(i + 1, n):
            if is_overlap(boxes[i], boxes[j]):
                ds.union(i, j)

    # Group boxes by disjoint set representative
    groups = {}
    for i in range(n):
        root = ds.find(i)
        if root not in groups:
            groups[root] = [i]
        else:
            groups[root].append(i)

    # Calculate merged rectangle for each group
    merged_boxes = []
    merged_contours = []
    for group in groups.values():
        min_x = min(boxes[i][0] for i in group)
        min_y = min(boxes[i][1] for i in group)
        max_x = max(boxes[i][0] + boxes[i][2] for i in group)
        max_y = max(boxes[i][1] + boxes[i][3] for i in group)
        merged_boxes.append((min_x, min_y,
                             max_x - min_x,
                             max_y - min_y))
        merged_contours.append([contours[i] for i in group])

    return merged_boxes, merged_contours


def format_contours(contours, boxes=None):
    assert(len(boxes) == len(contours))
    n = len(boxes)
    result = []
    for i in range(n):
        result.append({
            'contours': contours[i],
            'box': boxes[i]
        })

    return result


def get_areas(contours, width, height):
    width = int(width)
    height = int(height)
    area_list = []

    for cnt in contours:
        cnt = np.array([[point[0] * width,
                         point[1] * height]
                        for point in cnt], dtype=np.int32)
        area = cv2.contourArea(cnt)

        area_list.append(area / (width * height))

    return area_list


def get_image_size(image_path):
    """
    Returns the width and height of the image at the given path.
    
    :param image_path: Path to the image file.
    :return: A tuple (width, height) representing the dimensions of the image.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return (width, height)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def plot_confusion_matrix(y_true, y_pred, title, labels, xlabel, ylabel, percent=False, save_path=None):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    
        if percent:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2%"
        else:
            fmt = "d"
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        
        if save_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")  # Logging save path
        plt.close()  # Explicitly close the plot
    except ValueError as e:
        print(f"Error while plotting confusion matrix: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    return metrics


def process_pectoralis_bbox(input_224_3, input_256, model_pecto_cc, model_pectoralis_unet, threshold_class, threshold_seg, kernel_close, kernel_open , org_img_size, flip=False):
    result = {}

    # Predict class using model_pecto_cc - threshold fixed for 0.5
    pectoralis_class_bbox = _predict_class(model_pecto_cc, input_224_3, threshold=threshold_class, print_proba=False)

    # Predict overlays using model_pectoralis_unet
    overlays = {}
    overlays['pectoralis'] = _predict_overlays(input_256, model_pectoralis_unet, threshold=threshold_seg, closing_kernel=kernel_close, opening_kernel=kernel_open)[0]

    pectoralis = None
    if 'pectoralis' in overlays:
        pectoralis = find_contours(overlays['pectoralis'], flip=flip, original_img_size=org_img_size, biggest_only=True)
        if pectoralis:
            merged_boxes, merged_contours = merge_boxes(pectoralis['boxes'], pectoralis['contours'])
            result['pectoralis'] = format_contours(merged_contours, merged_boxes)
        else:
            result['pectoralis'] = None

    if result.get('pectoralis') is None:
        pectoralis_class_bbox = 'insufficient'

    return pectoralis_class_bbox


###########################################################################################################################################################################################

if __name__ == "__main__":

    # load csv 
    df = load_csv_as_dataframe(CSV_PATH)

    # load models
    model_pecto_cc = _load_one_model(MODELS_FOLDER, 'model_pecto_cc')
    custom_objects = {'iou_no_bcg': iou_no_bcg, 'iou': iou}
    model_pectoralis_unet = _load_one_model(MODELS_FOLDER, 'model_pectoralis_unet', custom_object=custom_objects)

    all_metrics = []


    for threshold_class in THRESHOLDS_CLASS:
        for threshold_seg in THRESHOLDS_SEG:
            for kernel_open in OPENING_KERNELS:
                for kernel_close  in CLOSING_KERNELS:


                    # for metrics
                    gt_pecto, pred_pecto, pred_original_pecto = [], [], []


                    # loop through all images
                    for image in tqdm(os.listdir(TESTSET_FOLDER), desc=f"Processing images with classification threshold {threshold_class}, segmentation threshold {threshold_seg}, opening_kernel {kernel_open} and closing kernel {kernel_close}"):

                        # to test
                        # if image != "1.2.840.113681.172503758.1720418990.11928.1686.1.png":
                        #     continue

                        # image info
                        img_path = os.path.join(TESTSET_FOLDER, image)
                        sop = image[:-4]
                        org_img_size = get_image_size(img_path)
                        flip = compare_image_sides(img_path) == "right"

                        # prepare image
                        img = load_img(img_path, color_mode="grayscale")
                        input_224_3 = _prepare_input_data(img, IMAGE_SIZE_224, flip, channels=3)
                        input_256 = _prepare_input_data(img, IMAGE_SIZE_256, flip, channels=1)

                        # extract information CSV
                        pectoralis_class_cordoba = df.loc[df['SOP Instance UID'] == sop, 'Pectoralis (CC) Cordoba'].iloc[0]
                        pectoralis_class_bbox_old = df.loc[df['SOP Instance UID'] == sop, 'Pectoralis (CC) b-box'].iloc[0]
                        gt_pecto.append(pectoralis_class_cordoba)
                        pred_original_pecto.append(pectoralis_class_bbox_old)

                        # predict bbox
                        pectoralis_class_bbox = process_pectoralis_bbox(input_224_3, input_256, model_pecto_cc, model_pectoralis_unet, threshold_class, threshold_seg, kernel_close, kernel_open, org_img_size, flip)
                        pred_pecto.append(pectoralis_class_bbox)

                        # print examples that are incorrect
                        # if pectoralis_class_bbox != pectoralis_class_bbox_old:
                        #     print(f'Problem with {sop}')
                        #     print('Overall: ', pectoralis_class_bbox)
                        #     print('CSV: ', pectoralis_class_bbox_old)



                    metrics = classification_metrics(gt_pecto, pred_pecto)
                    metrics["Threshold Classification"] = threshold_class
                    metrics["Threshold segmentation"] = threshold_seg
                    metrics["Opening Kernel"] = kernel_open
                    metrics["Closing Kernel"] = kernel_close
                    print(f"Metrics: ", metrics)
                    all_metrics.append(metrics)


                    plot_confusion_matrix(gt_pecto, pred_pecto, title="Confusion Matrix Pectoralis", labels=['correct', 'insufficient'], xlabel="b-box Prediction", ylabel="Cordoba GT", percent=False, save_path=f'plots/class_thresh_{threshold_class}_seg_thresh_{threshold_seg}_open_kern_{kernel_open}_close_kern_{kernel_close}/conf_matrix_bbox.png')
                    plot_confusion_matrix(gt_pecto, pred_pecto, title="Confusion Matrix Pectoralis (%)", labels=['correct', 'insufficient'], xlabel="b-box Prediction", ylabel="Cordoba GT", percent=True, save_path=f'plots/class_thresh_{threshold_class}_seg_thresh_{threshold_seg}_open_kern_{kernel_open}_close_kern_{kernel_close}/conf_matrix_bbox_percent.png')

    # original conf matrix only once
    plot_confusion_matrix(gt_pecto, pred_original_pecto, title="Confusion Matrix Pectoralis Original CSV", labels=['correct', 'insufficient'], xlabel="b-box CSV export", ylabel="Cordoba GT", percent=False, save_path=f'plots/original/conf_matrix_bbox_og.png')
    plot_confusion_matrix(gt_pecto, pred_original_pecto, title="Confusion Matrix Pectoralis Original CSV (%)", labels=['correct', 'insufficient'], xlabel="b-box CSV export", ylabel="Cordoba GT", percent=True, save_path='plots/original/conf_matrix_bbox_og_percent.png')

    metrics_original = classification_metrics(gt_pecto, pred_original_pecto)
    metrics_original["Threshold Classification"] = None
    metrics_original["Threshold segmentation"] = None
    metrics_original["Opening Kernel"] = None
    metrics_original["Closing Kernel"] = None
    all_metrics.append(metrics_original)
    print(f"Metrics original bbox predictions: ", metrics)


    # Save all metrics to a CSV file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
