import os
import json
import numpy as np
from matplotlib import pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    
    lines_hor = np.sort([box_1[0], box_1[2], box_2[0], box_2[2]])
    lines_ver = np.sort([box_1[1], box_1[3], box_2[1], box_2[3]])

    area_union = (lines_hor[3] - lines_hor[0]) * (lines_ver[3] - lines_ver[0])
    area_inter = (lines_hor[2] - lines_hor[1]) * (lines_ver[2] - lines_ver[1])


    iou = area_inter / area_union
        
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    TP_FP = 0
    TP_FN = 0
    record = []

    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        pred_thr = [item for item in pred if item[4]>=conf_thr]
        
        TP_FP += len(pred_thr)
        TP_FN += len(gt)
        
        if len(pred_thr) > 0:
        
            for i in range(len(gt)):
                iou_candidates = []
                for j in range(len(pred_thr)):          
                    
                    iou = compute_iou(pred_thr[j][:4], gt[i])
                    iou_candidates.append(iou)
                    
                iou_closest = max(iou_candidates)
                
                record.append([pred_file,i,iou_candidates.index(iou_closest)])
                
                if iou_closest >= iou_thr:
                    TP += 1 
             

    return TP, TP_FP, TP_FN, record


#%%

# set a path for predictions and annotations:
preds_path = '../results/hw02'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


#%%
# For a fixed IoU threshold, vary the confidence thresholds.

# The code below gives an example on the training set for one IoU threshold. 

confidence_thrs = np.sort(np.array([item[4] for fname in preds_train for item in preds_train[fname] if len(preds_train[fname])>0],dtype=float)) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
tp_fp_train = np.zeros(len(confidence_thrs))
tp_fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    if i == 0:
        tp_train[i], tp_fp_train[i], tp_fn_train[i], record = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)
    else:
        tp_train[i], tp_fp_train[i], tp_fn_train[i], _ = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves
# precision = 
# recall = 

fig,ax = plt.subplots()
ax.plot(tp_train/tp_fn_train, tp_train/tp_fp_train)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')



if done_tweaking:
    print('Code for plotting test set PR curves.')
    
    
    
    
    
    
