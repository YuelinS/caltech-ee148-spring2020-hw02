import os
import numpy as np
import json
from PIL import Image, ImageDraw 
from matplotlib import pyplot as plt


# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))

flag_figs = range(len(file_names_train))  #range(1)  #

flag_filter = range(2) # [0]  #

flag_heatmap = False #True #

# Red light filter parameter tuning:    
# filter shape params: (red_diameter, black_edge, blue_padding)
filter_paras = [(5,1),(15,2)]
filter_red = [450,350]  
filter_blk = [-90,-100] 

strides = [3,15]
thres = [1.4,13]

# normalize thre
norm_max = 200


drop_bottom = 75


# Set this parameter to True when you're done with algorithm development:
done_tweaking = True # False  #


#%%
def normalize(patch):
    # patch  = filter_pad

    # patch_zmean = patch - np.mean(patch)
    # patch_normed = patch_zmean / np.sqrt(np.sum(np.square(patch_zmean)))
    
    norm = np.amax(patch) - np.amin(patch) # max(, norm_max)
    patch_normed = (patch)/ norm  # - np.amin(patch)
                                        
    return patch_normed


def make_red_light_filter(filter_paras):
    
    filters = []

    for i in range(len(filter_paras)):
    
        # make the red light filter
        dia = filter_paras[i][0]
        edge = filter_paras[i][1]
        
        rad = int(np.floor(dia/2))        
        side = dia+edge*2
               
        filter_rgb = np.ones((side,side,3)) * filter_blk[i]
        
        circle_center = np.array((rad+edge,rad+edge))
        filter_circle_mask = np.array([[(np.sum((np.array((row,col))-circle_center)**2) <= rad**2) for col in range(side)] for row in range(side)])                                    
        filter_rgb[filter_circle_mask,0]= filter_red[i]

        filter_rgb = normalize(filter_rgb)

        plt.imshow(filter_rgb)
        plt.savefig(os.path.join('./red_light_filter_' + f'{i}' + '.png'))
        
        filters.append(filter_rgb)       
    
    return filters



def compute_convolution(I, filters):
    '''
    This function takes an image <I> and a filter <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''   

    convs = {}
    rows = {}
    cols = {}
        
    (im_rows,im_cols,im_channels) = np.shape(I)    
    
    
    for i in flag_filter: 
        
        filter_rgb = filters[i]
        (filter_rows,filter_cols,filter_channels) = np.shape(filter_rgb)
        stride = strides[i]
    
        i_convs = []
        i_rows = []
        i_cols = []
        for row in range(0,im_rows - filter_rows - drop_bottom,stride):
            
            for col in range(0,im_cols - filter_cols,stride):
    
                patch = I[row : row + filter_rows  , col : col + filter_cols]
    
                # normalize  
                corr = normalize(patch) * filter_rgb 
                
                i_convs.append(np.sum(corr))
                i_rows.append(row)
                i_cols.append(col)
                
        i_convs = np.array(i_convs)
        
        # draw heatmap
        if flag_heatmap:            
            conv_row = (im_rows - filter_rows - drop_bottom)//stride + int((im_rows - filter_rows - drop_bottom) % stride != 0)
            conv_col = (im_cols - filter_cols)//stride + int((im_cols - filter_cols) % stride != 0)
            heatmap = i_convs.reshape(conv_row,conv_col)
            
            plt.imshow(heatmap)
            plt.colorbar()
            
            # attach some information
            filter_range = heatmap.copy().flatten()
            filter_range.sort()
            plt.title(np.histogram(filter_range, bins=5)[1], size=10)
            plt.xlabel(filter_range[-10:][::-1])
            plt.show()
    
        convs[i] = i_convs
        rows[i] = i_rows
        cols[i] = i_cols
    
    return convs,rows,cols



def predict_boxes(convs,rows,cols):
    
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''    
    
    bboxes = [] # This should be a list of lists, each of length 4. 
    bbox_filter_idx = []
    top_convs = np.empty((10,4))
   
    
    for i in flag_filter:  
        
        filter_rgb = filters[i]
        (filter_rows,filter_cols,filter_channels) = np.shape(filter_rgb)
        
        thre = thres[i]  
        i_convs = convs[i]
        i_rows= rows[i]
        i_cols = cols[i]
        
        select_convs = i_convs > thre       
        select_rows = np.array(i_rows)[select_convs]
        select_cols = np.array(i_cols)[select_convs]
       
        pass_conv = []
        
        for j in range(len(select_rows)):
            
            if j not in pass_conv:
                conv = i_convs[select_convs][j]
                confidence = 1/(1 + np.exp(-conv)) 
                
                tl_row = int(select_rows[j])
                tl_col = int(select_cols[j])
                
                for k in range(len(select_rows)):
                    
                    tl_row2 = int(select_rows[k])
                    tl_col2 = int(select_cols[k])
                    
                    if k != j and abs(tl_row2 - tl_row)<filter_rows and abs(tl_row2 - tl_row)<filter_cols:
                        
                        pass_conv.append(k)
                        
                        conv2 = i_convs[select_convs][k]
                        confidence2 = 1/(1 + np.exp(-conv2))
                        
                        tl_row = (tl_row + tl_row2)/2 
                        tl_col = (tl_col + tl_col2)/2                       
                        confidence = (confidence + confidence2)/2                        
                
                br_row = tl_row + filter_rows
                br_col = tl_col + filter_cols
                bbox_filter_idx.append(i)
                bboxes.append([tl_row,tl_col,br_row,br_col,confidence]) 


    return bboxes, top_convs, bbox_filter_idx



def detect_red_light_mf(I,filters):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 
    '''
    

    convs,rows,cols = compute_convolution(I, filters)
    bboxes, top_convs, bbox_filter_idx = predict_boxes(convs,rows,cols)


    for i in range(len(bboxes)):
        assert len(bboxes[i]) == 5
        assert (bboxes[i][4] >= 0.0) and (bboxes[i][4] <= 1.0)

    return bboxes, top_convs, bbox_filter_idx



#%%

# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'
gts_path = '../data/hw02_annotations/'

# set a path for saving predictions:
preds_path = '../results/hw02'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# make filter
filters = make_red_light_filter(filter_paras)

#%%
'''
Make predictions on the training set.
'''

if not done_tweaking:

    # load annotation
    with open(os.path.join(gts_path,'annotations_train.json'),'r') as f:
        gts = json.load(f)
    
    
    preds_train = {}
    
    for i in flag_figs:
    
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))
        
        pred_bboxes, top_convs, bbox_filter_idx = detect_red_light_mf(np.asarray(I),filters)   
        preds_train[file_names_train[i]] = pred_bboxes
        
        gt_bboxes = gts[file_names_train[i]]   
        
        # visualization
        img = ImageDraw.Draw(I)  #Image.fromarray(I))  
        
        for bbox in gt_bboxes:
            bbox = [bbox[k] for k in [1,0,3,2]]
            img.rectangle(bbox)
        
        
        for bbox, filter_idx in zip(pred_bboxes, bbox_filter_idx):
            confidence = bbox[4]
            bbox = [bbox[k] for k in [1,0,3,2]]
            
            img.rectangle(bbox, outline = f"hsl({(filter_idx+1)*100}, 100%, 50%)")
            img.text([bbox[0]-10,bbox[1]-10],f'{confidence:.3f}')
            
        # I.show()
        I.save(os.path.join(preds_path,file_names_train[i]))
    
    
    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
        json.dump(preds_train,f)


#%%
'''
Make predictions on the test set. 
'''
if done_tweaking:
    
    # load
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))    
    
    with open(os.path.join(gts_path,'annotations_test.json'),'r') as f:
        gts = json.load(f)
        
    # predict    
    preds_test = {}
    for i in range(len(file_names_test)):

        I = Image.open(os.path.join(data_path,file_names_test[i]))
        preds_test[file_names_test[i]], _, _ = detect_red_light_mf(np.asarray(I),filters)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
