import json
import cv2
import os
"""
Annotations done with VGG Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/) 
and exported as coco json. Here transformed to detectron2 coco 

Make sure to change the image_parent_folder to your corresponding image directory
"""

file_to_transform = "./via_export_coco.json"
image_parent_folder = "./custom_data/train/"
only_one_class = False # in the publications we used only the single class "car"

json1_file = open(file_to_transform)
json1_str = json1_file.read()
data = json.loads(json1_str)


temp = []
for element in data['annotations']:
    element['image_id'] = int(element['image_id'])

    element['segmentation'] = [element['segmentation']]
    if only_one_class:
        element['category_id'] = 1
    temp.append(element)

data['annotations'] = temp

# modify if your categories are different
data["categories"] = [{
    "id": 1,
    "name": "car"
}, {
    "id": 2,
    "name": "truck"
}, {
    "id": 3,
    "name": "motorcycle"
}, {
    "id": 4,
    "name": "bicycle"
}, {
    "id": 5,
    "name": "pedestrian"
}, {
    "id": 6,
    "name": "camper"
}]

temp = []

for element in data['images']:

    im = cv2.imread(os.path.join(image_parent_folder + element['file_name']))
    shape = im.shape
    element['height'] = shape[0]
    element['width'] = shape[1]
    temp.append(element)
data['images'] = temp

with open('./transformed_annotations.json', 'w') as f:
    json.dump(data, f)
