import os
import json
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize
from PIL import Image, ImageDraw 


# Set this parameter to True when you're done with algorithm development:
done_tweaking = True # False #

flag_iou = [0.25, 0.5, 0.75]  #[0.25]   # 


def compute_iou(box1, box2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    # Determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
  
    interW = x2 - x1 +1
    interH = y2 - y1 +1
  
    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 :
      return - abs(box1[0] - box2[0])/100
  
    interArea = interW * interH
    box1Area = (box1[2] - box1[0] +1) * (box1[3] - box1[1] +1)
    box2Area = (box2[2] - box2[0] +1) * (box2[3] - box2[1] +1)
    iou = interArea / float(box1Area + box2Area - interArea)
           
    assert (iou >= 0) and (iou <= 1.0)

    return iou


# bbox_pred_thr, bbox_gt = preds_train['RL-001.jpg'], gts_train['RL-001.jpg']
# iou_thr=0.5
# conf_thr=0.5

def compute_counts(bbox_preds, bbox_gts, iou_thr=0.5, conf_thr=0.5):
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
    
    for pred_file, bbox_pred in bbox_preds.items():
        
    
        bbox_gt = bbox_gts[pred_file]
        bbox_pred_thr = [item for item in bbox_pred if item[4]>=conf_thr]
                      
        n_gt = len(bbox_gt)
        n_pred = len(bbox_pred_thr)
        
        TP_FP += n_pred
        TP_FN += n_gt
        
        # Find the best matching between gt and pred bbox
        if n_pred > 0 and n_gt > 0:
       
            MIN_IOU = 0.0
            
            # NUM_GT x NUM_PRED
            iou_matrix = np.zeros((n_gt, n_pred))
            for i in range(n_gt):
                for j in range(n_pred):
                    iou_matrix[i, j] = compute_iou(bbox_gt[i], bbox_pred_thr[j])
            
            if n_pred > n_gt:
              # there are more predictions than ground-truth - add dummy rows
              diff = n_pred - n_gt
              iou_matrix = np.concatenate((iou_matrix, 
                                           np.full((diff, n_pred), MIN_IOU)), 
                                           axis=0)
            
            if n_gt > n_pred:
              # more ground-truth than predictions - add dummy columns
              diff = n_gt - n_pred
              iou_matrix = np.concatenate((iou_matrix, 
                                           np.full((n_gt, diff), MIN_IOU)), 
                                           axis=1)
            
            # call the Hungarian matching
            idxs_gt, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
                    
            
            # remove dummy assignments and get the final match
            sel_actual = np.logical_and(idxs_pred < n_pred, idxs_gt < n_gt)
            idx_pred_actual = idxs_pred[sel_actual] 
            idx_gt_actual = idxs_gt[sel_actual]
            ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
            
            # decide positive / negative by iou threshold
            sel_valid = (ious_actual > iou_thr)
            TP += np.sum(sel_valid)
            
            
            # Visualize
            
            if int(pred_file[3:6])<20 and iou_thr==0.1 and conf_thr>0.80 and conf_thr<0.81:
                # read image using PIL:
                I = Image.open(os.path.join(data_path,pred_file))                  
                img = ImageDraw.Draw(I)  

                for count, idx in enumerate(idx_gt_actual):
                    bbox = [bbox_gt[idx][k] for k in [1,0,3,2]]
                    img.rectangle(bbox)
                    img.text([bbox[0]-10,bbox[1]-10],f'{count}')

            
                for count, idx in enumerate(idx_pred_actual):
                    bbox = [bbox_pred_thr[idx][k] for k in [1,0,3,2]]
                    img.rectangle(bbox, outline = "green")
                    img.text([bbox[2]-10,bbox[3]+10],f'{count}')

                    
                # I.show()
                I.save(os.path.join(preds_path, 'bbox_match_' + pred_file))
                                   

    return TP, TP_FP, TP_FN



#%%


# set a path for predictions and annotations:
data_path = '../data/RedLights2011_Medium'
preds_path = '../results/hw02'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))


'''
Load training data. 
'''

if not done_tweaking:

    with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
        preds_train = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_train = json.load(f)
    

    # For a fixed IoU threshold, vary the confidence thresholds.
    # training set for the three IoU threshold. 
    
    confidence_thrs = np.sort(np.array([item[4] for fname in preds_train for item in preds_train[fname] if len(preds_train[fname])>0],dtype=float)) # using (ascending) list of confidence scores as thresholds
    tp_train = np.zeros(len(confidence_thrs))
    tp_fp_train = np.zeros(len(confidence_thrs))
    tp_fn_train = np.zeros(len(confidence_thrs))
    
    # Plot training set PR curves
    # fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    for j, iou_thr in enumerate(flag_iou):
        
        for i, conf_thr in enumerate(confidence_thrs):
            tp_train[i], tp_fp_train[i], tp_fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)
    
        plt.plot(tp_train/tp_fn_train, tp_train/tp_fp_train,label = f'IoU threshold {iou_thr}')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Training set')    
    plt.legend()
    plt.savefig(os.path.join(preds_path,'PR_curve_train.png'))


#%%

if done_tweaking:
    
    '''
    Load test data.
    '''    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)
    
    tp_test = np.zeros(len(confidence_thrs))
    tp_fp_test = np.zeros(len(confidence_thrs))
    tp_fn_test = np.zeros(len(confidence_thrs))

    for j, iou_thr in enumerate(flag_iou):    
        
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], tp_fp_test[i], tp_fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr, conf_thr=conf_thr)
            
        plt.plot(tp_test/tp_fn_test, tp_test/tp_fp_test,label = f'IoU threshold {iou_thr}')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Test set')

    plt.savefig(os.path.join(preds_path,'PR_curve_test.png'))
    
    