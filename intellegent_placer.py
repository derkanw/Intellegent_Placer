import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from skimage.metrics import structural_similarity


class Config:
    objects_path = "Images\\Objects\\"
    background_path = "Images\\Background.jpg"
    ratio_thresh = 0.7
    objects_list = []
    rows = 2
    columns = 5


class Object:
    name = None
    points = None
    descriptors = None
    properties = None

    def __init__(self, name, points, descriptors, properties):
        self.name = name
        self.points = points
        self.descriptors = descriptors
        self.properties = properties

    def match_objects(self, recognized) -> float:
        object_size = len(self.points)
        recognized_size = len(recognized.points)
        if object_size < recognized_size:
            return recognized.match_objects(self)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(recognized.descriptors, self.descriptors, k=2)

        good = 0
        for m, n in matches:
            if m.distance > n.distance * Config.ratio_thresh:
                good += 1

        return good / object_size


def read_objects() -> None:
    titles, origin, images, cut_images = [], [], [], []

    for image_path in listdir(Config.objects_path):
        image = cv2.imread(path.join(Config.objects_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin_image = np.copy(image)

        # Selection of a part of the image along the contour of the object
        properties, extract_image = extract_object(image)
        image = np.array(properties.image, dtype=np.int32)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        points, descriptors = find_points(image)

        object_name, _ = image_path.split('.')
        Config.objects_list.append(Object(object_name, points, descriptors, properties))

        # The drawing part
        titles.append(object_name)
        images.append(extract_image)
        cut_images.append(image)
        origin.append(origin_image)

    draw_images(origin, titles, Config.rows, Config.columns)
    draw_images(images, titles, Config.rows, Config.columns)
    draw_images(cut_images, titles, Config.rows, Config.columns, True)


def extract_object(image: np.ndarray):
    # Finding areas of the image of an object and a sheet
    gray_image = rgb2gray(gaussian(image, sigma=3, channel_axis=True))
    otsu_image = threshold_otsu(gray_image)
    result_image = gray_image <= otsu_image
    result_image = binary_closing(result_image, footprint=np.ones((20, 20)))
    result_image = binary_opening(result_image, footprint=np.ones((10, 10)))

    # Getting object properties and applying mask
    labels = sk_measure_label(result_image)
    properties = regionprops(labels)
    item = np.array([element.area for element in properties]).argmin()
    mask = labels == (item + 1)
    image[~mask] = 255

    return properties[item], image


def find_points(image: np.ndarray):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)


def draw_images(images: list, names: list, rows: int, columns: int, gray=False) -> None:
    fig, axes = plt.subplots(rows, columns, figsize=(30, 30))

    for i in range(rows):
        for j in range(columns):
            index = i * columns + j
            axes[i, j].set_title(names[index])
            if gray:
                axes[i, j].imshow(images[index], cmap="gray")
            else:
                axes[i, j].imshow(images[index])
    plt.show()


def draw(image: np.ndarray, name: str, gray=False) -> None:
    if gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.title(name)
    plt.show()


def read_image(image_path: str) -> bool:
    # Getting the difference between background and objects
    background = cv2.cvtColor(cv2.imread(Config.background_path), cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    origin_image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, difference = structural_similarity(image, background, full=True)

    # Getting objects areas and a polygon
    difference = canny(difference, sigma=2.308, high_threshold=0.71)
    difference = binary_closing(difference, footprint=np.ones((5, 5)))
    difference = binary_fill_holes(difference)
    difference = binary_opening(difference, footprint=np.ones((15, 15)))

    # The drawing part
    draw(origin_image, "The input image")
    draw(difference, "Object areas on the input image")

    # Getting polygon properties
    labels = sk_measure_label(difference)
    properties = regionprops(labels)
    regions = np.array([element.centroid[0] for element in properties])
    polygon = regions.argmin()
    polygon_props = properties[polygon]

    # Recognition of known objects in a set
    objects_set = []
    objects_regions = [item for item in range(len(regions)) if item != polygon]
    for item in objects_regions:
        objects_set.append(find_object(properties[item]))

    # Trying to place objects in a polygon
    return place_objects(polygon_props, objects_set)


def find_object(region) -> Object:
    image = np.array(region.image, dtype=np.int32)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    points, descriptors = find_points(image)
    item = Object(None, points, descriptors, None)

    proportions = np.array([element.match_objects(item) for element in Config.objects_list])
    original_object = proportions.argmax()
    print(f"The matching object is {Config.objects_list[original_object].name} with value {np.max(proportions)}")

    return Config.objects_list[original_object]


def place_objects(polygon_prop, objects_set: list) -> bool:
    objects_area = np.sum([element.properties.area for element in objects_set])
    return objects_area < polygon_prop.area


def check_image(image_path: str) -> bool:
    read_objects()
    return read_image(image_path)
