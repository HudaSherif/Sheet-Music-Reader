
import numpy as np
import skimage.io as io
import math
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion,binary_opening, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
from commonfunctions import *
from skimage.transform import rotate
from skimage.filters import threshold_otsu

from skimage.filters import median, gaussian
from skimage.morphology.selem import disk
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction

from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode

from skimage.feature import hog


def normalize_image(img):
    max_img = np.max(img)
    return img / max_img

def adaptive_thresholding(img):
        M,N=img.shape
        image=np.copy(img)
        if(np.max(image)<=1):
            image=image*255

        no_ofpixels=M*N
        
        img_hist=getHist(image)
        showHist(image)
        Tinit=0
        for i in range(len(img_hist)):
                segma = i * img_hist[i]
                Tinit += segma
        
        Tinit=np.round(Tinit/no_ofpixels,decimals=0)
        t0=int(Tinit)
        t1=t0+1
        while(abs(t1 - t0) > 0.1):

            darker=img_hist[0:int(t0)]
            brighter=img_hist[int(t0):]
            dsum=np.sum(darker)
            bsum=np.sum(brighter)
            Tinitd=0
            Tinitb=0
            for i in range(len(darker)):
                Tinitd += i * darker[i]
            for j in range(len(brighter)):
                Tinitb +=(j+len(darker)) * brighter [j]
            davg=Tinitd/dsum
            bavg=Tinitb/bsum
            t1=t0
            t0=(davg+bavg)/2
        
        return t1



def otsu_adaptive(image):
    img=np.copy(image)

    told=0
    tnew=100
    while(abs(told-tnew)>0.05):
        told = tnew
        tnew=threshold_otsu(img)

        img_part1=img[img<tnew]
        img_part2=img[img>=tnew]

        mean1=img_part1.mean()
        mean2=img_part2.mean()

        img=img[(img>=mean1) & (img<=mean2)]

        
    return tnew

def deskew_img(binImg):

    negative_Image=1-binImg

    tested_angles = np.deg2rad(np.arange(0.1, 180.0,0.5))
    hspace, angles, dists= hough_line(negative_Image, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(hspace, angles, dists)

    
    most_common_angle = mode(np.around(angles, decimals=2))[0]
        
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle-(np.pi/2))

    rotatedImg=negative_Image
    if skew_angle>5 or skew_angle<-5: #deskewed picture
        if skew_angle<0 or (skew_angle>87 and skew_angle<93): #vertical
            skew_angle=180+skew_angle
        rotatedImg=rotate(negative_Image,skew_angle,mode='constant',resize=True)
        
    rotatedImg=normalize_image(rotatedImg)
    rotatedImg=rotatedImg>=0.95
    return 1-rotatedImg



def initial_removalOfStaves(img):

    m,n=img.shape
    blackrun_verticies=[]
    max_whiteruns=[]
    max_blackruns=[]
    x_count_dict={}
    for i in range(n):
        current_vertexpair=[]

        current_sequence=[]
        current_element=1
        counter=0
        for j in range(m):
            if(j==0): #bn3mlha 3shan dymn bnbd2 b sequence el wa7ayd f incase el sora kant bd2a b 0
                if(img[j,i]==0):
                    current_sequence.append(0)
                    current_element=0

                    current_vertexpair.append((j,i))

            if(img[j,i]==current_element):
                counter+=1

                if(current_element==0):
                    if not(j in x_count_dict.keys()):
                        x_count_dict[j] = 0
                    x_count_dict[j]=x_count_dict[j]+1

            else:
                current_sequence.append(counter)
                counter=1
                current_element=1-current_element
                if(current_element==1):
                    current_vertexpair.append((j-1,i))
                    blackrun_verticies.append(current_vertexpair)
                    current_vertexpair=[]

                else:
                    current_vertexpair.append((j,i))
                    if not(j in x_count_dict.keys()):
                        x_count_dict[j] = 0
                    x_count_dict[j]=x_count_dict[j]+1

        if(current_element==0):
            current_vertexpair.append((m-1,i))
            blackrun_verticies.append(current_vertexpair)
                


        current_sequence.append(counter)
        white_runs=np.array(current_sequence[::2])
        black_runs=np.array(current_sequence[1::2])
        
        staff_space=np.bincount(white_runs).argmax()
        if(black_runs.size!=0):
            staff_thickness=np.bincount(black_runs).argmax()
        else:
            staff_thickness=0

        max_blackruns.append(staff_thickness)
        max_whiteruns.append(staff_space)

    max_blackruns=np.array(max_blackruns)
    temp=np.array(max_whiteruns)
    max_whiteruns=temp[temp<1/4*img.shape[0]]
    max_blackruns=max_blackruns[max_blackruns>0]
    return np.bincount(max_whiteruns).argmax(),np.bincount(max_blackruns).argmax(),blackrun_verticies,x_count_dict



def remove_staffLines(binary_img,black_run_verticies,staff_line_positions,staff_thickness):
    phase1_img=np.copy(binary_img)
    #if thickness is far away from staff line thickness the it is not a staff line and remove it
    for vertex_pair in black_run_verticies:
        x1,y1=vertex_pair[0]
        x2,y2=vertex_pair[1]
        distance=x2-x1  #as we only consider vertical thickness
        # mean=(x1+x2)/2
        if not(int(x1) in staff_line_positions) or not(int(x2) in staff_line_positions): #checking if it was one the suspected rows
           continue
        if(distance>=(staff_thickness-2) and distance<=(staff_thickness+2)):
            #if(binary_img[x1-1,y1]==1 or binary_img[x2+1,y1]==1):
            phase1_img[x1:x2+1,y1]=1 #removing the staff line
    
    return phase1_img

     



#this function returns a dictionary enumrating the staff lines and each staff line points at the pixels it represent
def get_staffline_dictionary(staff_line_positions, staff_space, staff_thickness):
    
    x={}

    x[1]=[staff_line_positions[0]]  ## awl flag byshawr 3la awl mkan 

    current_staffLine=1
    counter=0
    for index in staff_line_positions[1:]:
        if(x[current_staffLine][counter]+(staff_space/2)>index):
            x[current_staffLine].append(index)
            counter+=1
        else:
            current_staffLine+=1
            counter=0
            x[current_staffLine]=[index]



    if(len(x.keys())%5==0):
        return x
    
    size=len(x.keys())
    x_list=list(x.keys())
    x_list.sort()
    if(len(x.keys())<5):
        while(len(x.keys())<5):
            current_key=size
            x[current_key+1]=[x[current_key][-1]+staff_space+i for i in range(0, staff_thickness)]
            size+=1

    else:
        delete_keys=[]
        dummy_list=[]
        current_List=[]

        counter=0

        #current_List.append(1)
        for key,value in x.items():
            if len(current_List)==0:
                current_List.append(key)
                continue
            
            
            if max(x[key-1])+(1.5*staff_space)>min(value)  and len(current_List)<5:
                current_List.append(key)

            else:
                dummy_list.append(current_List)
                current_List=[key]
    
        
        for l in dummy_list:           
            if len(l)<5:
                
                for key_inner in l:
                    del x[key_inner]



    return x




#segmenting symbols from an image
def segment_image(image):
    
    img=np.copy(image)
    contours=find_contours(img, 0.8)
    contours=np.array(contours)
    bounding_boxes=[]
    cutoff_size = 80
    for contour in contours:
        ymax=np.max(contour[:,0])
        ymin=np.min(contour[:,0])
        xmax=np.max(contour[:,1])
        xmin=np.min(contour[:,1])
        temp=[int(xmin),int(xmax),int(ymin),int(ymax)]
        contour_size = abs(xmax - xmin) * abs(ymax - ymin)
        bounding_boxes
        if contour_size < cutoff_size:
            continue
        
        should_continue = False
        for i_box, box in enumerate(bounding_boxes): ##checking if abounding box is contained in another bounding box
            x_b1, x_b2, y_b1, y_b2 = box
            is_new_in_old = (xmin >= x_b1 and xmax <= x_b2 and ymin >= y_b1 and ymax <= y_b2)
            is_old_in_new = (x_b1 >= xmin and x_b2 <= xmax and y_b1 >= ymin and y_b2 <= ymax)
            if is_new_in_old:
                should_continue = True
                break
            elif is_old_in_new:
                should_continue = True
                bounding_boxes[i_box] = [int(x_b1),int(x_b2),int(y_b1),int(y_b2)]
                break
            
        if not should_continue:
            bounding_boxes.append(temp)
        
    img_with_boxes=np.array(img)
    
    img_segmented_list = []
    xmin_dict = {}#this is used to be able to sort shapes
    xmin_list = []
    ymin_dict = {}
    
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        rr, cc = rectangle(start = (Ymin,Xmin), end = (Ymax,Xmax), shape=img.shape)
        img_with_boxes[rr, cc] = 0
        xmin_dict[Xmin] = img[rr, cc].T
        xmin_list.append(Xmin)
        ymin_dict[Xmin] = Ymin
    
    xmin_list.sort()
    ymin_list = [] ##this is used to detect the minimum y of each shape
    for xmin in xmin_list:
        img_segmented_list.append(xmin_dict[xmin])
        ymin_list.append(ymin_dict[xmin])


    return img_segmented_list, img_with_boxes, ymin_list, xmin_list



def relative_headposition(box_position,staff_line_dict, staff_space, staff_thickness):
    
    mean_values = []
    for v in staff_line_dict.values():
        mean_values.append(sum(v) / len(v))

    mids =[]# [mean_values[0] - threshold]
    for mean_i in range(0, len(mean_values)-1):
        mean_1 = mean_values[mean_i]    
        mean_2 = mean_values[mean_i+1]
        mid = (mean_1 + mean_2) / 2
        mids.append(mid)
    
    
    boundary_indices = mean_values + mids
    boundary_indices.sort()
    first_element=boundary_indices[0]
    last_element=boundary_indices[-1]

    before_elements=[first_element-staff_space-staff_thickness-(staff_space/2),first_element-(staff_space/2)-staff_thickness,first_element-(staff_space/2)]
    after_elements=[last_element+(staff_space/2),last_element+staff_space+(staff_thickness/2)]

    boundary_indices=before_elements+boundary_indices+after_elements
    boundary_indices=np.array(boundary_indices)
    distances = abs(boundary_indices - box_position)

    min_distance_i = np.argmin(distances)
    
    return min_distance_i



def get_blackpix(img):
    h,w=img.shape
    
    blackpixels=0
    for i in range(h):
        for j in range(w):
            if img[i][j]==0:

                blackpixels+=1
    
    return blackpixels/(h*w)



def get_orientation(img):
    h,w=img.shape
    
    blackpixelsup=0
    blackpixlesdown=0
    orient=""


    rows = img.shape[0]

    upper_sum = np.sum(img[:int(rows/2), :])
    lower_sum = np.sum(img[int(rows/2):, :])

    if lower_sum < upper_sum: # ta7t feh black aktar
        return 'heads-down'
    else:
        return 'heads-up'
 




def get_vertical_lines(img, staff_line_space, foreground=255):
    img = normalize_image(img)
    img = 1 - img
    img = thin(img, max_iter=1000)
    img = img.astype('uint8')
    img = img * 255

    vertical_line_positions = []
    threshold = 2*staff_line_space
    last_detected_column = -10
    column_threshold = 5
    for j in range(img.shape[1]):
        max_black_run = 0
        current_black_run = 0
        max_black_run_end_pos = (0, j)
        for i in range(img.shape[0]):
            if img[i, j] == foreground:
                current_black_run += 1
            elif current_black_run >= 2:
                if (j+1 < img.shape[1] and img[i, j+1] == foreground) or (j+2 < img.shape[1] and img[i, j+2] == foreground) or (j-1 >= 0 and img[i, j-1] == foreground) or (j-2 >= 0 and img[i, j-2] == foreground):
                    current_black_run += 1
                else:
                    if current_black_run > max_black_run:
                        max_black_run = current_black_run
                        max_black_run_end_pos = (i-1, j)
                    current_black_run = 0
            else:
                if current_black_run > max_black_run:
                    max_black_run = current_black_run
                    max_black_run_end_pos = (i-1, j)
                current_black_run = 0
        
        if current_black_run > max_black_run:
            max_black_run = current_black_run
            max_black_run_end_pos = (i-1, j)

        if max_black_run > threshold and (j - last_detected_column) > column_threshold:
            vertical_line_positions.append((max_black_run_end_pos[0] - max_black_run + 1, j))
            last_detected_column = j

        
    return len(vertical_line_positions), vertical_line_positions



def get_horizontal_lines(img, staff_line_space, foreground=255):
    
    img = normalize_image(img)
    img = 1 - img
    img = thin(img, max_iter=1000)
    img = img.astype('uint8')
    img = img * 255

    horizontal_line_positions = []
    threshold = 2.5*staff_line_space
    last_detected_row = -10
    row_threshold = 2
    invalid_rows = set()
    for i in range(img.shape[0]):
        max_black_run = 0
        current_black_run = 0
        starting_i = i
        starting_j = 0
        current_invalid_rows = set()
        max_invalid_rows = set()
        if i in invalid_rows:
            continue
        for j in range(img.shape[1]):
            # i+1 j-1, i j-1, i-1 j-1, i-1 j
            if current_black_run == 0 and ( (i+1 < img.shape[0] and j-1 >= 0 and img[i+1, j-1] == foreground) or  (j-1 >= 0 and img[i, j-1] == foreground) or (i-1 >= 0 and j-1 >= 0 and img[i-1, j-1] == foreground) ):
                continue
            if img[i, j] == foreground:
                current_black_run += 1
                current_invalid_rows.add(i)
            elif (i-1 >=0 and img[i-1, j] == foreground) and current_black_run >= 1 and not (i-1 in invalid_rows):
                i -= 1
                current_black_run += 1
                current_invalid_rows.add(i)
            elif (i+1 < img.shape[0] and img[i+1, j] == foreground) and current_black_run >= 1 and not (i+1 in invalid_rows):
                i += 1
                current_black_run += 1
                current_invalid_rows.add(i)
            else:
                if current_black_run > max_black_run:
                    max_black_run = current_black_run
                    starting_j = j - max_black_run
                    max_invalid_rows = set(current_invalid_rows)
                current_black_run = 0
                current_invalid_rows = set()
                i = starting_i
        
        if current_black_run > max_black_run:
            max_black_run = current_black_run
            starting_j = j - max_black_run
            max_invalid_rows = set(current_invalid_rows)
            i = starting_i

        if max_black_run > threshold and (i - last_detected_row) > row_threshold:
            horizontal_line_positions.append((starting_i, starting_j))
            last_detected_row = i
            invalid_rows = invalid_rows.union(max_invalid_rows)


    return len(horizontal_line_positions), horizontal_line_positions


def get_number_of_flags(img):
    
    img = normalize_image(img)
    orientation = get_orientation(img)
    
    new_image = img.copy() #thin(img, max_iter=1000)
    no_of_rows = new_image.shape[0]
    no_of_removed_rows = int(no_of_rows * 0.28)
    
    new_image = thin(1-new_image, max_iter=1000)
    if orientation == 'heads-up':
        new_image = new_image[no_of_removed_rows:, :]
    else:
        new_image = new_image[:-no_of_removed_rows, :]
    
    threshold = int(img.shape[1] * 0.12)
    
    current_white_run = 0
    white_runs_dict = {}
    for j in range(new_image.shape[1]):
        no_of_white = len(np.where(new_image[:, j] == 1)[0])
        
        if no_of_white > 8:
            continue
        if not(no_of_white in white_runs_dict.keys()):
            white_runs_dict[no_of_white] = 0
        
        white_runs_dict[no_of_white] += 1

    max_candidate = 0
    if 0 in white_runs_dict.keys():
        del white_runs_dict[0]
    dict_keys = list(white_runs_dict.keys())
    
    cumulative_sum = 0
    for key in dict_keys:
        cumulative_sum = white_runs_dict[key]
        if cumulative_sum >= threshold and key > max_candidate:
            max_candidate = key
    
    return max_candidate



def black_heads(img, radius = 4, mode=0):
    
    image = np.copy(img)

    if mode == 0:
        image=binary_dilation(image,disk(3))
    else:
        image=binary_dilation(image,disk(4))
        image=binary_dilation(image,np.ones((1,12)))


    padded_column = np.ones((image.shape[0], 1))

    image = np.concatenate((image, padded_column), axis=1)
    image = np.concatenate((padded_column, image), axis=1)

    padded_row = np.ones((1, image.shape[1]))
    image = np.concatenate((image, padded_row), axis=0)
    image = np.concatenate((padded_row, image), axis=0)
    
    boxes=[]

    se = disk(radius)

    output1 = binary_dilation(image,se)
    output2 = binary_erosion(output1,se)

    label_img = label(output2, background=-1 ,connectivity=2)
    regions = regionprops(label_img)

    count = 0
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        should_continue = False
        for props_inner in regions:
            minr_inner, minc_inner, maxr_inner, maxc_inner = props_inner.bbox
            if maxr == maxr_inner and minr == minr_inner and maxc == maxc_inner and minc == minc_inner:
                continue
            if minc_inner >= minc and maxc_inner <= maxc and minr_inner >= minr and maxr_inner <= maxr:
                should_continue = True
                break
        
        if should_continue:
            continue

        aspect_ratio = (maxr - minr)/ (maxc-minc)
        
        if aspect_ratio > 0.5 and aspect_ratio < 1.3 :   
            count  = count + 1
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            boxes.append([int(minc), int(maxc), int(minr), int(maxr)])
    
    return bx, by,count,boxes



def white_heads(img, higher_aspect_ratio=1):
    
    image = np.copy(img)
    padded_column = np.ones((image.shape[0], 1))

    image = np.concatenate((image, padded_column), axis=1)
    image = np.concatenate((padded_column, image), axis=1)

    padded_row = np.ones((1, image.shape[1]))
    image = np.concatenate((image, padded_row), axis=0)
    image = np.concatenate((padded_row, image), axis=0)


    se = np.array([[0,0,1,1,1],
                   [0,1,1,1,1],
                   [1,1,1,1,0],
                   [1,0,0,0,0]])
    img1 = binary_erosion(image, se)
    img1 = binary_opening(img1, se)
    img1 = binary_dilation(img1, se)

    seed = np.copy(img1)
    seed[1:-1, 1:-1] = img1.min()

    mask = img1
    dilated = reconstruction(seed, mask, method='dilation')
    
    
######################################

##################################

    img2 = img1 - dilated
    img2 = 1-img2

    label_img = label(img2, background=-1 ,connectivity=2)
    regions = regionprops(label_img)

    if np.max(img2) == np.min(img2):
        return [], 0
    
##################################
################################## 

    count = 0
    bounding_boxes = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        aspect_ratio = (maxr - minr)/ (maxc-minc)

        if aspect_ratio > 0.5 and aspect_ratio < higher_aspect_ratio:
            count  = count + 1
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)

            bounding_boxes.append([int(minc), int(maxc), int(minr), int(maxr)])
                
##################    
    return   bounding_boxes,count




def get_staff_width(img):
    col1 = 0
    err_margin = 10
    for col_i in range(img.shape[1]):
        col = img[:, col_i]
        if np.sum(col) < (img.shape[0] - err_margin):
            col1 = col_i
            break

    col2 = 0
    for col_i in list(range(img.shape[1]))[::-1]:
        col = img[:, col_i]
        if np.sum(col) < (img.shape[0] - err_margin):
            col2 = col_i
            break
    
    return col2-col1



def get_staffline_positions(img, threshold, foreground=255):
    
    img = normalize_image(img)
    img = 1 - img
    img = img.astype('uint8')
    img = img * 255

    vertical_line_positions = []

    last_detected_row = -10
    row_threshold = 0

    x_list = []
    for i in range(img.shape[0]):
        max_black_run = 0
        current_black_run = 0
        max_black_run_end_pos = (i, 0)
        starting_i = i
        for j in range(img.shape[1]):
            if img[i, j] == foreground:
                current_black_run += 1
            elif current_black_run >= 2:
                if i+1 < img.shape[0] and img[i+1, j] == foreground:
                    current_black_run += 1
                elif i-1 >= 0 and img[i-1, j] == foreground:
                    current_black_run += 1
                else:
                    if current_black_run > max_black_run:
                        max_black_run = current_black_run
                        max_black_run_end_pos = (i-1, j)
                    current_black_run = 0
            else:
                if current_black_run > max_black_run:
                    max_black_run = current_black_run
                    max_black_run_end_pos = (starting_i, j-1)
                current_black_run = 0
        
        if current_black_run > max_black_run:
            max_black_run = current_black_run
            max_black_run_end_pos = (starting_i, j)
        
        i = starting_i
        if max_black_run > threshold and (i - last_detected_row) > row_threshold:
            vertical_line_positions.append((i, max_black_run_end_pos[1] - max_black_run + 1))
            last_detected_row = i
            x_list.append(i)
        
    return  x_list



def get_staffline_positions_modified(img, threshold, foreground=255):
    
    img = normalize_image(img)
    img = 1 - img

    img = img.astype('uint8')
    img = img * 255

    vertical_line_positions = []

    last_detected_row = -10
    row_threshold = 0
    
    row_offset_threshold = 8

    x_list = []
    for i in range(img.shape[0]):
        max_black_run = 0
        current_black_run = 0
        max_black_run_end_pos = (i, 0)
        starting_i = i

        for j in range(img.shape[1]):
            if img[i, j] == foreground:
                current_black_run += 1
            elif current_black_run >= 2 and (abs(i - starting_i) + 1) < row_offset_threshold:
                if i+1 < img.shape[0] and img[i+1, j] == foreground: # and not(i+1 in invalid_x_set)
                    current_black_run += 1
                    i += 1
                elif i-1 >= 0 and img[i-1, j] == foreground: #and not(i-1 in invalid_x_set)
                    current_black_run += 1
                    i -= 1
                else:
                    if current_black_run > max_black_run:
                        max_black_run = current_black_run
                        max_black_run_end_pos = (i-1, j)
                    current_black_run = 0
            else:
                if current_black_run > max_black_run:
                    max_black_run = current_black_run
                    max_black_run_end_pos = (starting_i, j-1)
                current_black_run = 0
        
        if current_black_run > max_black_run:
            max_black_run = current_black_run
            max_black_run_end_pos = (starting_i, j)
        
        i = starting_i

        if max_black_run > threshold and (i - last_detected_row) > row_threshold:
            vertical_line_positions.append((i, max_black_run_end_pos[1] - max_black_run + 1))
            last_detected_row = i
            x_list.append(i)
        
    return  x_list



def get_vertical_lines_acc(img, threshold, foreground=1):
    vertical_line_positions = []
    last_detected_column = -10
    column_threshold = 5
    for j in range(img.shape[1]):
        max_black_run = 0
        current_black_run = 0
        max_black_run_end_pos = (0, j)
        for i in range(img.shape[0]):
            if img[i, j] == foreground:
                current_black_run += 1
            elif current_black_run >= 2:
                if (j+1 < img.shape[1] and img[i, j+1] == foreground) or (j+2 < img.shape[1] and img[i, j+2] == foreground) or (j-1 >= 0 and img[i, j-1] == foreground) or (j-2 >= 0 and img[i, j-2] == foreground):
                    current_black_run += 1
                else:
                    if current_black_run > max_black_run:
                        max_black_run = current_black_run
                        max_black_run_end_pos = (i-1, j)
                    current_black_run = 0
            else:
                if current_black_run > max_black_run:
                    max_black_run = current_black_run
                    max_black_run_end_pos = (i-1, j)
                current_black_run = 0
        
        if current_black_run > max_black_run:
            max_black_run = current_black_run
            max_black_run_end_pos = (i-1, j)
        if max_black_run > threshold and (j - last_detected_column) > column_threshold:
            vertical_line_positions.append((max_black_run_end_pos[0] - max_black_run + 1, j))
            last_detected_column = j
        
    return len(vertical_line_positions), vertical_line_positions



def get_horizontal_lines_acc(img, threshold, foreground=1):
    horizontal_line_positions = []
    last_detected_row = -10
    row_threshold = 2

    for i in range(img.shape[0]):
        max_black_run = 0
        current_black_run = 0
        starting_i = i

        for j in range(img.shape[1]):

            if current_black_run == 0 and ( (i+1 < img.shape[0] and j-1 >= 0 and img[i+1, j-1] == foreground) or  (j-1 >= 0 and img[i, j-1] == foreground) or (i-1 >= 0 and j-1 >= 0 and img[i-1, j-1] == foreground) ):
                i = starting_i
                continue
            if img[i, j] == foreground:
                current_black_run += 1
            elif (i-1 >=0 and img[i-1, j] == foreground) and current_black_run >= 1:# and not (i-1 in invalid_rows):
                current_black_run += 1
            elif (i+1 < img.shape[0] and img[i+1, j] == foreground) and current_black_run >= 1:# and not (i+1 in invalid_rows):
                current_black_run += 1
            else:
                if current_black_run > max_black_run:
                    max_black_run = current_black_run
                    starting_j = j - max_black_run
                current_black_run = 0
        
        if current_black_run > max_black_run:
            max_black_run = current_black_run
            starting_j = j - max_black_run

        if max_black_run > threshold and (i - last_detected_row) > row_threshold:
            horizontal_line_positions.append((starting_i, starting_j))
            last_detected_row = i

    return len(horizontal_line_positions), horizontal_line_positions

def hashtag_vs_naruto(img):
 
    img = normalize_image(img)
    img[img >= 0.5] = 1
    img[img < 0.5] = 0

    img = 1 - img
    img = thin(img, max_iter=1000)
    img = img.astype('int')

    horizontal_lines = get_horizontal_lines_acc(img, threshold=img.shape[1] * 0.12)[0]
    vertical_lines = get_vertical_lines_acc(img, threshold=img.shape[0] * 0.25)[0]

    if horizontal_lines != 2 or vertical_lines != 2:
        return 'unknown'
    
    image_rows = img.shape[0]
    top_image = img[:int(image_rows*0.3), :]

    top_vertical_lines = get_vertical_lines_acc(top_image, threshold=image_rows*0.3*0.85)[0]
    if top_vertical_lines == 2:
        return 'hashtag'
    elif top_vertical_lines == 1:
        return 'naruto'
    else:
        return 'unknown'

def is_acc_cross(img):
    img = normalize_image(img)
    img[img >= 0.5] = 1
    img[img < 0.5] = 0

    img = 1 - img
    img = thin(img, max_iter=1000)
    img = img.astype('uint8')

    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    hspace, angles, dists = hough_line(img, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(hspace, angles, dists)
    
    angles = angles * 180 / np.pi
    angles = angles.astype('uint8')
    
    accum_copy = np.array(accum).copy()

    max_i1 = np.argmax(accum_copy)
    max_accum1 = accum_copy[max_i1]
    accum_copy[max_i1] = 0
    
    max_i2 = np.argmax(accum_copy)
    max_accum2 = accum_copy[max_i2]
    
    found_large_angle = False # found an angle between 127 - 140
    found_small_angle = False # found an angle between 37 - 50
    for angle_i, angle in enumerate(angles):
        if accum[angle_i] != max_accum1 and accum[angle_i] != max_accum2:
            continue
        if 127 <= angle <= 140:
            found_large_angle = True
        elif 37 <= angle <= 50:
            found_small_angle = True

    if found_small_angle and found_large_angle:
        return 'cross'
    else:
        return 'unknown'


def hashtag_vs_naruto_vs_cross(img):
    classif = hashtag_vs_naruto(img)
    if classif != 'unknown':
        return classif
    
    classif = is_acc_cross(img)
    return classif





def classify_img(img, ymin, staff_space, staff_thickness, staffline_collection, head_letter_dict, prev_output=''):
    output = ""
    if(img.shape[0]>3.5*(staff_thickness+staff_space)): ##notes with stemss
        if( get_blackpix(img)>0.95):
            output=""
        else:
            count1=0
            count2=0

            by1 = None
            boxes1 = None

            by2 = None
            boxes2 = None

            try:
                _,by1,count1,boxes1 = black_heads(img, radius= 4, mode=0)
            except:
                pass

            try:
                _,by2,count2,boxes2 = black_heads(img, radius= 4, mode=1)
            except:
                pass

            if count1 >= count2 or count2 >= 4:
                count = count1
                by = by1
                boxes = boxes1
            else:
                count = count2
                by = by2
                boxes = boxes2

            if(count==0):#hole
                boxes,count=white_heads(img)
                if(count>0):
                    by_min=boxes[0][2]
                    by_max=boxes[0][3]
                    box_y_position=int((by_min+by_max)/2)
                    box_y_position+=ymin
                    output=head_letter_dict[relative_headposition(box_y_position, staffline_collection, staff_space, staff_thickness)]+"/2"
                else:
                    output=""
                    return ''
                # pass

            elif(count==1):# /4   or flag
                no_of_flags=get_number_of_flags(img)
                
                if(no_of_flags==0):
                    by_min=np.min(by)
                    by_max=np.max(by)
                    box_y_position=int((by_min+by_max)/2)
                    box_y_position+=ymin
                    output=head_letter_dict[relative_headposition(box_y_position,staffline_collection, staff_space, staff_thickness)]+"/4"
                else:
                    by_min=np.min(by)
                    by_max=np.max(by)
                    box_y_position=int((by_min+by_max)/2)
                    box_y_position+=ymin

                    output=head_letter_dict[relative_headposition(box_y_position,staffline_collection, staff_space, staff_thickness)]+"/"+f"{4*np.power(2,no_of_flags)}"

            else: #chords or beams
                no_of_stems,_=get_vertical_lines(img,staff_space)
                if(no_of_stems==1): #chord
                    output="{"
                    output_list=[]
                    for box in boxes:
                        by_min=box[2]
                        by_max=box[3]
                        box_y_position=int((by_min+by_max)/2)
                        box_y_position+=ymin
                        hpos = relative_headposition(box_y_position,staffline_collection, staff_space, staff_thickness)
                        output_list.append(head_letter_dict[hpos]+"/4")
                    
                    output_list.sort()
                    
                    output += ", ".join(output_list)
                    output+="}"  
                
                else:#beam
                    output="beam"
                    h_number, h_number_list=get_horizontal_lines(img,staff_space)
                    
                    output=""
                    boxes_to_output_dict={}
                    x_min_list=[]
                    for box in boxes:
                        xmin=box[0]
                        by_min=box[2]
                        by_max=box[3]
                        box_y_position=int((by_min+by_max)/2)
                        box_y_position+=ymin
                        output = head_letter_dict[relative_headposition(box_y_position,staffline_collection, staff_space, staff_thickness)]+"/"+f"{4*np.power(2,h_number)}"+" "
                        boxes_to_output_dict[xmin] = output
                        x_min_list.append(xmin)


                    #to sort according their arrangment fromleft to right
                    x_min_list.sort()
                    output=""
                    for x_min in x_min_list:
                        output += boxes_to_output_dict[x_min]
                
    else: #accendintals or notehead or dots

        if get_blackpix(img) > 0.7:
            # black dot
            return '.'
        
        img_cols = img.shape[1]
        large_vertical_lines, vertical_line_positions1 = get_vertical_lines(img[:, int(img_cols * 0.5):], img.shape[0] * 0.35 * 0.5) # * 0.5 because it is multiplied by 2 in the function
        large_vertical_lines2, _ = get_vertical_lines(img[:, :int(img_cols * 0.5)], img.shape[0] * 0.35 * 0.5) # * 0.5 because it is multiplied by 2 in the function

        white_boxes,white_count=white_heads(img)
    
        large_vertical_lines = large_vertical_lines + large_vertical_lines2
        
        aspect_ratio = img.shape[1] / img.shape[0]
        if large_vertical_lines == 1:
            if aspect_ratio >= 1:
                y_position = img.shape[0] / 2
                y_position += ymin
                return head_letter_dict[relative_headposition(y_position,staffline_collection, staff_space, staff_thickness)] + '/1'
            return '&'
        elif large_vertical_lines > 1: # bb, naruto OR hashtag
            output = ''
            upper_large_vertical_lines, _ = get_vertical_lines(img[:int(img.shape[0] * 0.4), :], img.shape[0] * 0.1 * 0.7 * 0.5)
            if upper_large_vertical_lines == 1: # naruto
                return ''
            elif upper_large_vertical_lines == 0:
                y_position = img.shape[0] / 2
                y_position += ymin
                return head_letter_dict[relative_headposition(y_position,staffline_collection, staff_space, staff_thickness)] + '/1'
            
            # bb OR hashtag
            upper_horizontal_lines, upper_horizontal_positions = get_horizontal_lines(img[:int(img.shape[0] * 0.3), :], img.shape[1] * 0.5 * 0.5)
            if upper_horizontal_lines >= 1:
                # hashtag
                return output + '#'
            else:
                return '&'*2 #large_vertical_lines
        
        cross_classif = is_acc_cross(img)
        if cross_classif == 'cross':
            return '##'
        
        vertical_lines,_= get_vertical_lines(img[:int(img.shape[0] * 0.3)], img.shape[0] * 0.5 * 0.7 * 0.5)

        if vertical_lines >= 1:
            # naruto
            return ''

        y_position = img.shape[0] / 2
        y_position += ymin
        return head_letter_dict[relative_headposition(y_position,staffline_collection, staff_space, staff_thickness)] + '/1'                
    return output





def classify_number(img):

    thinned_img = thin(1-img, max_iter=1000)

    right_image = thinned_img[int(thinned_img.shape[0] * 0.4):, int(thinned_img.shape[1] * 0.4):]

    vertical_lines, _ = get_vertical_lines_acc(right_image, 0.5 * right_image.shape[0] * 0.4)

    if vertical_lines >= 1:
        return "4"
    
    return "2"


#classification

def classify_all(imgsegements_list, head_letter_dict, staff_space, staff_thickness,stafflines_dictionary_list, outFile):
    output_str = ""
    if len(imgsegements_list) > 1:
        output_str += "{\n"
    for line_number, img_list in enumerate(imgsegements_list):

        img_segmented_list, img_with_boxes, ymin_list, xmin_list = segment_image(img_list)

        img_segmented_list_filtered = list(img_segmented_list)
        ymin_list_filtered = list(ymin_list)
        xmin_list_filtered = list(xmin_list)

        if len(img_segmented_list_filtered) == 0:
            continue

        while len(img_segmented_list_filtered) > 0 and get_blackpix(img_segmented_list_filtered[0]) > 0.45:
            img_segmented_list_filtered = img_segmented_list_filtered[1:]
            ymin_list_filtered = ymin_list_filtered[1:]
            xmin_list_filtered = xmin_list_filtered[1:]

        if len(img_segmented_list_filtered) == 0:
            continue

        img_segmented_list_filtered = img_segmented_list_filtered[1:]
        ymin_list_filtered = ymin_list_filtered[1:]
        xmin_list_filtered = xmin_list_filtered[1:]

        if len(img_segmented_list_filtered) == 0:
            continue
        
        output_str += "[ "
        if len(img_segmented_list_filtered) > 1:
        # try:
            ymin1 = ymin_list_filtered[0]
            ymin2 = ymin_list_filtered[1]

            xmin1 = xmin_list_filtered[0]
            xmin2 = xmin_list_filtered[1]

            if abs(xmin2 - xmin1) <= 10:

                segmented_image_1 = img_segmented_list_filtered[0]
                segmented_image_2 = img_segmented_list_filtered[1]
                
                ymax1 = ymin1 + segmented_image_1.shape[0]
                ymax2 = ymin2 + segmented_image_2.shape[0]

                if ymin2 > ymax1:
                    number_output = '\meter<"4/'
                    number_output += classify_number(segmented_image_2)
                    number_output += '">'
                    img_segmented_list_filtered = img_segmented_list_filtered[2:]
                    ymin_list_filtered = ymin_list_filtered[2:]
                    
                    output_str += number_output + " "
                elif ymin1 > ymax2:
                    number_output = '\meter<"4/'
                    number_output += classify_number(segmented_image_1)
                    number_output += '">'
                    img_segmented_list_filtered = img_segmented_list_filtered[2:]
                    ymin_list_filtered = ymin_list_filtered[2:]
                    output_str += number_output + " "
                    
                elif ymin1 == ymin2 and ymax2 == ymax2:
                    img_segmented_list_filtered = img_segmented_list_filtered[2:]
                    ymin_list_filtered = ymin_list_filtered[2:]
                    output_str += '\meter<"4/4">' + " "

        prev_output = ''
        for segment_number, segmented_image in enumerate(img_segmented_list_filtered):
            
            output = ""
            should_print = True
            try:
                current_output = classify_img(segmented_image, ymin_list_filtered[segment_number], staff_space, staff_thickness, stafflines_dictionary_list[line_number], head_letter_dict, '')
                if current_output == '' or str(current_output).find('&') != -1 or str(current_output).find('#') != -1:
                    should_print = False
                    prev_output += current_output
                else:
                    output = current_output[0] + prev_output + current_output[1:]
                    prev_output = ''
            except:
               pass
            
            if output != "" and should_print:
                output_str += output + " "
        
        output_str += "]\n"
    
    if len(imgsegements_list) > 1:
        output_str += "}\n"
    
    
    try:
        with open(outFile, 'x') as f:
            f.write(output_str)
            f.close()
    except:
        with open(outFile, 'w') as f:
            f.write(output_str)
            f.close()



def generate_output_file(inFile, outFile):
    head_letter_dict={} #according to the symbol position relative to staff line we assign a character

    head_letter_dict[0]='b2'
    head_letter_dict[1]='a2'
    head_letter_dict[2]='g2'
    head_letter_dict[3]='f2'
    head_letter_dict[4]='e2'
    head_letter_dict[5]='d2'
    head_letter_dict[6]='c2'
    head_letter_dict[7]='b1'
    head_letter_dict[8]='a1'
    head_letter_dict[9]='g1'
    head_letter_dict[10]='f1'
    head_letter_dict[11]='e1'
    head_letter_dict[12]='d1'
    head_letter_dict[13]='c1'

    img=rgb2gray(io.imread(inFile))

    threshold_img=gaussian(img,1.6)
    threshold = otsu_adaptive(threshold_img)

    binary_img=img>threshold
    binary_img = deskew_img(binary_img)

    staff_space,staff_thickness,black_run_verticies,x_count=initial_removalOfStaves(binary_img)  #black run verticies represents vertex pairs (starting and ending) of the black run

    staff_line_positions = []
    staff_width = get_staff_width(binary_img)

    staff_line_positions=get_staffline_positions(binary_opening(binary_img,np.ones((4,1))), 0.25*staff_width, foreground=255)

    staff_line_positions.sort()
    img_staffline_removed = remove_staffLines(binary_img,black_run_verticies,staff_line_positions,staff_thickness)

    img_staffline_removed=1-img_staffline_removed

    img_staffline_removed=binary_erosion(img_staffline_removed,selem=np.ones((int(1.6*staff_thickness), 1)))
    img_staffline_removed=binary_dilation(img_staffline_removed,selem=np.ones((int(1.65*staff_thickness), 1)))
    img_staffline_removed=binary_dilation(img_staffline_removed,selem=np.ones((1, int(1.25*staff_thickness))))
    img_staffline_removed=binary_erosion(img_staffline_removed,np.ones((2,3)))

    img_staffline_removed=1-img_staffline_removed

    staff_line_dictionary = get_staffline_dictionary(staff_line_positions, staff_space, staff_thickness)
    
    img_staffline_removed=median(img_staffline_removed)


    #segmenting the image into segments containing each staffline collection
    number_of_stafflines=len(staff_line_dictionary.keys())
    number_of_collections=(int)(number_of_stafflines/5)

    stafflines_dictionary_list=[]
    imgsegements_list=[]
    for i in range(number_of_collections):
        stafflines_dictionary_list.append(dict(list(staff_line_dictionary.items())[(i*5):(i*5)+5]))
        thres=3*(staff_space+staff_thickness)
        
        min_image_position=min(list(stafflines_dictionary_list[i].values())[0])-thres
        max_image_position=max(list(stafflines_dictionary_list[i].values())[4])+thres

        if(min_image_position<0):
            min_image_position=0
        if(max_image_position>binary_img.shape[0]):
            max_image_position=binary_img.shape[0]


        #adjusting the staff lines positions to their relative collection of the image
        for key,value in stafflines_dictionary_list[i].items():
            v=np.array(value)
            stafflines_dictionary_list[i][key]=v-min_image_position

        #clipping the picture
        imgsegements_list.append(img_staffline_removed[min_image_position:max_image_position])

    

    #print(number_of_collections,stafflines_dictionary_list)
    classify_all(imgsegements_list, head_letter_dict, staff_space, staff_thickness,stafflines_dictionary_list, outFile)




    
        














