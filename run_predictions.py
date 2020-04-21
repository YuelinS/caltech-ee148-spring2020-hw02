import os
import numpy as np
import json
from PIL import Image, ImageDraw 
from matplotlib import pyplot as plt



# Red light filter parameter tuning:
    
# filter shape params: (red_diameter, black_edge, blue_padding)
filter_paras = [(7,2,0),(7,2,3),(23,3,0),(23,3,5)]
strides = [5,5,10,10]
thres = [0,5000,0,10000]
   
# filter color params:
k = 140
sky = [150-k,180-k,200-k]
# sky = [100, 200, 255]

filter_w_red = 400
filter_w_blk_upper, filter_w_blk_lower = -60, -60

# normalize thre
norm_max = 200

# using filter #:
flag_filter = range(4) # [0]  #

drop_bottom = 72


def normalize(patch):
    # patch  = filter_pad

    # patch_zmean = patch - np.mean(patch)
    # patch_normed = patch_zmean / np.sqrt(np.sum(np.square(patch_zmean)))
    
    norm = max(np.amax(patch) - np.amin(patch), norm_max)
    patch_normed = (patch)/ norm  # - np.amin(patch)
                                        
    return patch_normed


def make_red_light_filter(filter_paras):
    
    filters = []

    for i in range(len(filter_paras)):
    
        # make the red light filter
        dia = filter_paras[i][0]
        edge = filter_paras[i][1]
        pad = filter_paras[i][2]
        
        rad = int(np.floor(dia/2))
        
        filter_rows = dia*3+edge
        filter_cols = dia+edge*2
        
        circle_center = np.array((rad+edge,rad+edge))
        
        filter_rgb_upper = np.ones((dia,filter_cols,3)) * filter_w_blk_upper
        filter_rgb_lower = np.ones((dia*2+edge,filter_cols,3)) * filter_w_blk_lower
        filter_rgb = np.vstack([filter_rgb_upper, filter_rgb_lower])

        filter_circle_mask = np.array([[(np.sum((np.array((row,col))-circle_center)**2) <= rad**2) for col in range(filter_cols)] for row in range(filter_rows)])     
        
        # filter_rgb = np.ones((dia+edge*2,dia+edge*2,3))*(-10)
        # filter_circle_mask = np.array([[(np.sum((np.array((row,col))-circle_center)**2) <= rad**2) for col in range(dia)] for row in range(dia)])     
                     
        
        filter_rgb[filter_circle_mask,0]= filter_w_red
        
        filter_pad = np.empty((filter_rows+pad*2,filter_cols+pad*2,3))
        
        for channel in range(3):
        
            filter_pad[:,:,channel] = np.pad(filter_rgb[:,:,channel], (pad,pad), 'constant', constant_values=[(sky[channel],sky[channel])])
        
        
        # smooth
        
        # make red light filter by image
        # I = Image.open(os.path.join(data_path,file_names[1]))
        # I.show()
        filter_pad = normalize(filter_pad)

        plt.imshow(filter_pad)
        # plt.show()
        plt.savefig(os.path.join('./red_light_filter_' + f'{i}' + '.png'))
    
    
        filters.append(filter_pad)
        
    
    return filters



def compute_convolution(I_array, filter_rgb, stride=None):
    '''
    This function takes an image <I> and a filter <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''   
    
    (im_rows,im_cols,im_channels) = np.shape(I_array)

    # for i in flag_filter:      
        # print(f'filter_{i}')
        
    (filter_rows,filter_cols,filter_channels) = np.shape(filter_rgb)

    stride = strides[0]
    
    convs = []
    rows = []
    cols = []
    
    for row in range(0,im_rows - filter_rows - drop_bottom,stride):
        
        for col in range(0,im_cols - filter_cols,stride):
    
                patch = I_array[row : row + filter_rows  , col : col + filter_cols]
    
                # normalize  
                corr = normalize(patch) * filter_rgb 
                
                convs.append(np.sum(corr))
                rows.append(row)
                cols.append(col)
    
    # draw heatmap
    conv_row = (im_rows - filter_rows - drop_bottom)//stride + int((im_rows - filter_rows - drop_bottom) % stride != 0)
    conv_col = (im_cols - filter_cols)//stride + int((im_cols - filter_cols) % stride != 0)
    heatmap = np.array(convs).reshape(conv_row,conv_col)
    
    plt.imshow(heatmap)
    plt.colorbar()
    rgb_range = heatmap.copy().flatten()
    rgb_range.sort()
    plt.title(np.histogram(rgb_range, bins=5)[1], size=10)
    plt.xlabel(rgb_range[-10:][::-1])
    plt.show()

    convs = np.array(convs)
    
    return convs,rows,cols



def predict_boxes(convs,rows,cols):
    
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    # bounding_box_filter_idx = []
    top_convs = np.empty((10,4))
    thre = thres[0]


    # sort_convs = np.sort(convs)[::-1]
    # sort_convs_idx = np.argsort(convs)[::-1]
    # top_convs = sort_convs[:10]
    
    select_convs = convs > thre       
    # assert sum(select_convs) > 10    
    select_rows = np.array(rows)[select_convs]
    select_cols = np.array(cols)[select_convs]
   

    for j in range(len(select_rows)):
        
        conv1 = convs[select_convs][j]*5
        confidence = 1/(1 + np.exp(-conv1)) 
        
        tl_row = int(select_rows[j])
        tl_col = int(select_cols[j])
        br_row = tl_row + filter_rows
        br_col = tl_col + filter_cols
        # bounding_box_filter_idx.append(i)
        bounding_boxes.append([tl_row,tl_col,br_row,br_col,confidence]) 


    return bounding_boxes, top_convs   #, bounding_box_filter_idx



def detect_red_light_mf(I_array,filter_rgb):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    

    convs,rows,cols = compute_convolution(I_array, filter_rgb)
    bounding_boxes, top_convs = predict_boxes(convs,rows,cols)


    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 5
        assert (bounding_boxes[i][4] >= 0.0) and (bounding_boxes[i][4] <= 1.0)

    return bounding_boxes, top_convs



#%%

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'
gts_path = '../data/hw02_annotations/'


# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../results/hw02'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False


'''
Make predictions on the training set.
'''

flag_figs = range(16) #range(len(file_names_train))  #

# load annotation
with open(os.path.join(gts_path,'annotations_train.json'),'r') as f:
    gts = json.load(f)
    

# make filter
filters = make_red_light_filter(filter_paras)
filter_rgb = filters[0]
(filter_rows,filter_cols,filter_channels) = np.shape(filter_rgb)


preds_train = {}
for i in flag_figs:

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I_array = np.asarray(I)

    bounding_boxes, top_convs = detect_red_light_mf(I_array,filter_rgb)
    
    preds_train[file_names_train[i]] = bounding_boxes
    
    gt_bounding_boxes = gts[file_names_train[i]]
    

    # visualization
    img = ImageDraw.Draw(I)  #Image.fromarray(I))  

    for bounding_box in gt_bounding_boxes:
        bounding_box = [bounding_box[k] for k in [1,0,3,2]]
        img.rectangle(bounding_box)


    for bounding_box in bounding_boxes:
        bounding_box = [bounding_box[k] for k in [1,0,3,2]]
        img.rectangle(bounding_box, outline ="red")
        
    # I.show()
    I.save(os.path.join(preds_path,file_names_train[i]))
    
    
# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)


if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
