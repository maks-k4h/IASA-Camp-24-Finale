import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import shutil
import json
from typing import Union, Any, Tuple, List, Dict


CLASS_TO_INT = dict()


class UIElement:
    role: str
    x1: int
    y1: int
    x2: int
    y2: int

    children: List["UIElement"]

    visible: bool

    def __init__(self, role: str, bbox: Tuple[int], visible: bool) -> None:
        """
        bbox: (x1, y1, x2, y2)
        """

        self.role = role
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.children = []
        self.visible = visible

    
def json_to_tree(dict: Dict) -> UIElement:
    role = dict['role']
    if 'visible_bbox' in dict and dict['visible_bbox'] is not None:
        x, y, w, h = dict['visible_bbox']
        element = UIElement(role, (x, y, x+w, y+h), visible=True)
    else:
        x, y, w, h = dict['bbox']
        element = UIElement(role, (x, y, x+w, y+h), visible=False)
    for child_dict in dict['children']:
        if child_dict['bbox'] is not None:
            element.children.append(json_to_tree(child_dict))
    return element


def export_tree_to_yolov8(root: UIElement, output_file_p: Path) -> None:
    def to_lines(node: UIElement, window_w: int, window_h: int) -> List[str]:
        if node.visible:
            node_class_idx = CLASS_TO_INT[node.role]
            node_x_center = (node.x1 + node.x2) / 2 / window_w
            node_y_center = (node.y1 + node.y2) / 2 / window_h
            node_width = (node.x2 - node.x1) / window_w
            node_height = (node.y2 - node.y1) / window_h

            line = f"{node_class_idx} {node_x_center} {node_y_center} {node_width} {node_height}"
            lines = [line]
        else:
            lines = []
        for child in node.children:
            lines.extend(to_lines(child, window_w=window_w, window_h=window_h))
        
        return lines
    
    window_w = root.x2
    window_h = root.y2
    lines = to_lines(root, window_h=window_h, window_w=window_w)[1:]  # exclude window element
    
    with open(output_file_p, 'w') as f:
        f.writelines(l + '\n' for l in lines)

    


def update_classes_with_tree(root: UIElement) -> None:
    global CLASS_TO_INT
    if not root.role in CLASS_TO_INT:
        CLASS_TO_INT[root.role] = len(CLASS_TO_INT)
    for child in root.children:
        update_classes_with_tree(child)



def run(
        input_p: Path,
        output_p: Path
) -> None:
    labels_p = output_p / 'labels'
    train_p = output_p / 'train'

    os.makedirs(labels_p, exist_ok=True)
    os.makedirs(train_p, exist_ok=True)

    for app_dir in input_p.iterdir():
        if not app_dir.is_dir():
            continue
        for screen_dir in app_dir.iterdir():
            if not screen_dir.is_dir():
                continue
            image_p = None
            json_p = None

            for sub in screen_dir.iterdir():
                if sub.suffix in {'.jpg', '.jpeg', '.png'}:
                    image_p = sub
                if sub.suffix == '.json':
                    json_p = sub
            
            if json_p is None or image_p is None:
                print(screen_dir, '- skipping')
                continue
            
            with open(str(json_p), 'r') as f:
                meta = json.load(f)

            print(json_p)
            root = json_to_tree(meta)
            update_classes_with_tree(root)

            stem = f"{app_dir.name}-{screen_dir.name}"
            export_tree_to_yolov8(root, labels_p / f"{stem}.txt")
            shutil.copy(image_p, train_p / f"{stem}{image_p.suffix}")

    # other metadata
    dataset_meta_str = ""
    dataset_meta_str += f"train: {train_p.name}\n"
    dataset_meta_str += f"nc: {len(CLASS_TO_INT)}\n"
    INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}
    classes_list = list(INT_TO_CLASS[i] for i in range(len(INT_TO_CLASS)))
    dataset_meta_str += f"names: {classes_list}\n"
    with open(output_p / 'data.yaml', 'w') as f:
        f.write(dataset_meta_str)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', action='store', help='path to hackathon data (directory with apps directories)', required=True)
    parser.add_argument('--output', action='store', help='directory where to put dataset', required=True)
    
    args = parser.parse_args()

    input_p = Path(args.input)
    output_p = Path(args.output)

    run(
        input_p=input_p,
        output_p=output_p
    )
