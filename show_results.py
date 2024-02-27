import matplotlib.pyplot as plt
import numpy as np
import json

def print_things(models_names, models, title) :
    task_names = ["result_mask_MAP", "result_bbox_MAP"]

    mAPs = {
        'result_mask_MAP': {
            'mAP': [],
            'mAP_50': [],
            'mAP_75': [],
        }, 'result_bbox_MAP': {
            'mAP': [],
            'mAP_50': [],
            'mAP_75': [],
        }
    }

    for task in task_names :
        for model in models_names :
            mAPs[task]['mAP'].append(json_object[model]["result_mask_MAP"]["map"])
            mAPs[task]['mAP_50'].append(json_object[model]["result_mask_MAP"]["map_50"])
            mAPs[task]['mAP_75'].append(json_object[model]["result_mask_MAP"]["map_75"])

    x = np.arange(len(models)) # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in mAPs['result_bbox_MAP'].items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_xticks(x + width, models)
    ax.legend(ncols=3)
    ax.set_ylim(0, 0.35)

    plt.show()


with open('output/results.json', 'r') as json_file:
    json_object = json.load(json_file)

############ 1° CONFRONTO ##############
"""
    Confronto del modello allenato con Accessory rispetto al modello
    senza Accessory e modello con Custom Loss
    Tutti allenati su 4 epochs con bs di 8
"""

title = "Comparison of Initial Models"
models_names = ["no_pretrain_accessory", "no_pretrain_no_accessory", "no_pretrain_custom_loss"]
models = ("No Accessory", "Accessory", "Custom Loss")

print_things(models_names, models, title)

############ 2° CONFRONTO ##############
"""
    Confronto il modello allenato con Accessory per 12 epochs con SGD
    il modello allenato con Accessory per 12 epochs con Adam (lr=0.0005)
"""

title = "Comparison of Accessory Model with SGD or Adam"
models_names = ["accessory_12_epochs", "Adam"]
models = ("SGD", "Adam")

print_things(models_names, models, title)

############ 3° CONFRONTO ##############
"""
    Confronto del modello migliore del 2° confronto, quindi quasi sicuramente SGD
    con il modello utilizzando la versione 2 della mask-rcnn di Pytorch, su 12 epochs
"""

title = "Comparison of Accessory Model using V1 or V2 (PyTorch)"
models_names = ["accessory_12_epochs", "V2"]
models = ("V1", "V2")

print_things(models_names, models, title)