import json
from PIL import Image
from pathlib import Path

fish_categories = {'ALB':0, 'BET':1, 'DOL':2, 'LAG':3, 'OTHER':5, 'SHARK':6, 'YFT':7}
for fish in fish_categories:
    # Load labels for each fish category
    with open(f'./datasets/labels/{fish}.json') as f:
        img_labels = json.load(f)
    
    for img in img_labels:
        img_name = Path(img['filename']).stem
        img_path = f'./datasets/images/{img["filename"]}'
        img_width, img_height = Image.open(img_path).size
        
        annotations_list = []
        for a in img['annotations']:
            # class x_center y_center width height
            annotations_list.append([fish_categories[a['class']], round(a['x']/img_width, 6), round(a['y']/img_height, 6), 
                      round(a['width']/img_width, 6), round(a['height']/img_height, 6)])

        with open(f"./datasets/labels/{fish}/{img_name}.txt", "a") as f:
            for item in annotations_list:
                f.write(" ".join([str(i) for i in item]) + "\n")
