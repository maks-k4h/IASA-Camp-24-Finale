import os
import sys
from typing import List, Any, Tuple, Dict
from pathlib import Path
from argparse import ArgumentParser
import json
from json import JSONEncoder
from functools import cmp_to_key
from os import makedirs

import numpy as np

import cv2
from ultralytics import YOLO

import easyocr


IMAGE_SUFFS = {'.jpeg', '.jpg', '.png', '.webp'}

ocr_reader = easyocr.Reader(['en', 'uk'])
gui_model = None


def init_gui_model(yolo_model_p: Path) -> None:
    global gui_model
    gui_model = YOLO(yolo_model_p.__str__())


# apply super-intelligent algorithm to remove the three circles at the top
# todo: keypoint matching? blob detection?
def fancy_control_buttons_remover(macos_screenshot: np.ndarray) -> np.ndarray:
    image = np.copy(macos_screenshot)
    patch_color = image[10:50, 150:200].mean(axis=(0, 1))
    image[10:50, 10:150] = patch_color
    return image


class UIElement:
    role: str
    role_description: str
    text: str

    bbox: List[int]

    children: List["UIElement"]

    def __init__(
            self,
            bbox: List[int], 
            role: str = "",
            role_description: str = "",
            text: str = "",
    ) -> None:
        self.role = role
        self.role_description = role_description
        self.text = text
        self.bbox = [int(e) for e in bbox]
        self.children = []

    def __del__(self) -> None:
        for child in self.children:
            del child

    def sort(self) -> None:
        for child in self.children:
            child.sort()
        self.children.sort(key=cmp_to_key(self._node_comparator))

    @staticmethod
    def _node_comparator(a: "UIElement", b: "UIElement") -> int:
        if a.bbox[1] != b.bbox[1]:
            return 1 if a.bbox[1] > b.bbox[1] else -1
        else:
            return 1 if a.bbox[0] > b.bbox[0] else -1

    def add_element(self, element: "UIElement") -> None:
        assert self._try_add_element(element), "The element must be within the parent's bbox."

    def _try_add_element(self, element: "UIElement") -> bool:
        # check if the element is within the bbox
        if not (element.bbox[0] >= self.bbox[0] and
                element.bbox[1] >= self.bbox[1] and
                element.bbox[2] <= self.bbox[2] and
                element.bbox[3] <= self.bbox[3]):
            return False
        # try to add the element to the children if applicable
        if 'text' not in self.role.lower() and 'image' not in self.role.lower():
            for child in self.children:
                if child._try_add_element(element):
                    return True
        # append element as a first-order child
        self.children.append(element)
        self.role_description = 'group'
        return True

    @property
    def __dict__(self) -> Dict:
        return {
            'role': self.role,
            'role_description': self.role_description,
            'text': self.text,
            'bbox': [self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]],
            'children': self.children
        }


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def get_bboxes(image_p: Path) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    return: list of gui elements' classes with their bounding boxes
    """
    result = model.predict(image_p.__str__(), conf=0.02, iou=0.1)[0]

    res = [(result.names[int(cls_int)], (x1, y1, x2, y2)) for cls_int, (x1, y1, x2, y2) in zip(result.boxes.cls, result.boxes.xyxy)]

    return res


def get_image_w_h(image_p: Path) -> Tuple[int, int]:
    image = cv2.imread(image_p.__str__())
    return image.shape[1], image.shape[0]


def ocr_fill(image: np.ndarray, root: UIElement) -> None:
    if 'text' in root.role.lower() or 'button' in root.role.lower():
        ocr_res = ocr_reader.readtext(image[root.bbox[1]:root.bbox[3], root.bbox[0]:root.bbox[2]])
        root.text = ' '.join(text for bbox, text, prob in ocr_res)
    for child in root.children:
        ocr_fill(image, child)


def get_ui_tree(image_p: Path) -> UIElement:
    res = get_bboxes(image_p)
    w, h = get_image_w_h(image_p)
    root = UIElement(role='AXWindow', role_description='group', bbox=[0, 0, w, h])
    for cls, bbox in res:
        node = UIElement(role=cls, bbox=bbox)
        root.add_element(node)

    ocr_fill(fancy_control_buttons_remover(cv2.imread(image_p.__str__())), root)

    root.sort()

    return root


def process_recursively(data_p: Path, output_p: Path, folder_hierarchy: List[str] = []) -> None:
    current_path = data_p.joinpath(*folder_hierarchy)
    assert current_path.is_dir(), current_path.__str__()

    for sub in current_path.iterdir():
        if sub.is_file() and sub.suffix in IMAGE_SUFFS:
            dir_out_p = output_p.joinpath(*folder_hierarchy)
            makedirs(dir_out_p, exist_ok=True)
            json_out_p = dir_out_p / 'my_elemenets_v3.json'
            if json_out_p.exists():
                print('Skipping', '/'.join(folder_hierarchy))
                continue
            print('Processing', '/'.join(folder_hierarchy))

            # process image
            tree = get_ui_tree(sub)
            
            # save json
            with open(json_out_p.__str__(), 'w') as f:
                json.dump(tree, f, cls=MyEncoder, indent=4)

            del tree

        if sub.is_dir():
            process_recursively(data_p, output_p, folder_hierarchy + [sub.name])


def run(
        data_p: Path,
        output_p: Path,
        yolo_model_p: Path,
) -> None:
    for p in [data_p]:
        assert p.exists(), str(p)
    os.makedirs(output_p, exist_ok=True)

    init_gui_model(yolo_model_p)

    process_recursively(data_p=data_p, output_p=output_p)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input', action='store', required=True, help='path to data')
    parser.add_argument('--output', action='store', required=True, help='directory where to put results')

    parser.add_argument('--model', action='store', required=False,
                        help='path to yolo model')

    args = parser.parse_args()

    run(
        data_p=Path(args.input),
        output_p=Path(args.output),
        roboflow_api_key=args.rb_api_key
    )
    

