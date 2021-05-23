import cv2
import numpy as np

##################################################### REQUIRED METHODS ########################################################

# array to store the images:
ImgDataset=[]

#loading the dataset iamges from specified folder:
for i in range(351,408):  
    image=cv2.imread("/img_dataset/DSC_0"+str(i)+".JPG")  
    ImgDataset.append(image)

#intrinsic camera params matrix K:
K = np.array([ [2393.952166119461, -3.410605131648481e-13, 932.3821770809047], 
               [0, 2398.118540286656, 628.2649953288065], 
               [0, 0, 1]
            ])

#if we scale_down the images then we also need to scale down the K matrix
scale_down_factor=2.0
K[0,0]=K[0,0]/scale_down_factor
K[1,1]=K[1,1]/scale_down_factor
K[0,2]=K[0,2]/scale_down_factor
K[1,2]=K[1,2]/scale_down_factor

#Function to scale-down the images:
def scale_down_image(img,scale_down_factor):
    scale_down_factor=scale_down_factor/2
    i=0
    while(i<scale_down_factor):
        img=cv2.pyrDown(img)
        i+=1
    
    return img

#Function to get feature exctraction and matching for 2 images: 
def feature_extraction_and_matching(img1, img2):
    #turning RGB images to grey-scale images:
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    #SIFT for feature extraction: 
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1=sift.detectAndCompute(img1,None)
    kp2,des2=sift.detectAndCompute(img2,None)
    des1=np.float32(des1)
    des2=np.float32(des2)
    
    #FLANN for featuer matching:
    FLANN_INDEX_KDTREE = 1
    idx_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(idx_params,search_params)
    match = flann.knnMatch(des1,des2,k=2)
    
    #Lowe's Distance ratio test:
    good_matches = []
    for m,n in match:
        if m.distance<0.7*n.distance:
            good_matches.append(m)
           
    kp1=np.float32([kp1[m.queryIdx].pt for m in good_matches])
    kp2=np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    return kp1,kp2

#Function to triangulate and get the 3D points:
def triangulate(proj_prev,proj_next,kp1,kp2):
    kp1=kp1.T
    kp2=kp2.T
    point_cloud = cv2.triangulatePoints(proj_prev,proj_next,kp1,kp2)
    point_cloud/=point_cloud[3]
    
    return kp1,kp2,point_cloud

#Function of perspective-n-point method
def p_n_p(point_cloud,kp1,kp2,K): 
    distortion_coeff = np.zeros((5,1))
    _,rot,trans,inliers = cv2.solvePnPRansac(point_cloud,kp1,K,distortion_coeff,cv2.SOLVEPNP_ITERATIVE)
    inliers = inliers[:,0]
    rot,_ = cv2.Rodrigues(rot)

    return rot, trans, kp2[inliers], point_cloud[inliers], kp1[inliers]

#Function to calculate the reprojection error of 3d point cloud campared to image plane
def get_reproj_err(point_cloud,kp,trans_matrix,K):
    rot,_ = cv2.Rodrigues(trans_matrix[:3,:3]) 
    kp_reprojected,_ = cv2.projectPoints(point_cloud,rot,trans_matrix[:3,3],K,distCoeffs=None)
    kp_reprojected = np.float32(kp_reprojected[:,0,:]) 
    err = cv2.norm(kp_reprojected,kp,cv2.NORM_L2)
    
    return err/len(kp_reprojected), point_cloud, kp_reprojected

#Function to return common points in the 3 images and a set of non-common points
def trifocal_view(kp2,kp2_next,kp3):
    """
    Kp2 are keypoints we got from image(n-1) and image(n)
    kp2_next and kp3 are the keypoints we got from image(n) and image(n+1)
    """
    idx1=[]
    idx2=[]
    for i in range(kp2.shape[0]):
        if (kp2[i,:]==kp2_next).any():
            idx1.append(i)
        x=np.where(kp2_next==kp2[i,:])
        if x[0].size!=0:
            idx2.append(x[0][0])
    
    # we find non-common keypoints as well:
    kp3_uncommon=[]
    kp2_next_uncommon=[]
    for k in range(kp3.shape[0]):
        if k not in idx2:
            kp3_uncommon.append(list(kp3[k,:]))
            kp2_next_uncommon.append(list(kp2_next[k,:]))
    
    idx1=np.array(idx1)
    idx2=np.array(idx2)
    kp2_next_common=kp2_next[idx2]
    kp3_common=kp3[idx2]
            
    return idx1, kp2_next_common, kp3_common, np.array(kp2_next_uncommon), np.array(kp3_uncommon)
 
# Function to generate an output.ply 3D object file that can be rendered in MeshLab:       
def output(final_point_cloud,point_colours):
    
    output_points = final_point_cloud.reshape(-1, 3)*200
    output_colors = point_colours.reshape(-1, 3)
    mesh = np.hstack([output_points,output_colors])
    mesh_mean = np.mean(mesh[:,:3],axis=0)
    diff = mesh[:,:3]-mesh_mean
    distance = np.sqrt(diff[:,0]**2+diff[:,1]**2+diff[:,2]**2) 
    index = np.where(distance<np.mean(distance)+300)
    mesh = mesh[index]

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    with open('/output.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(mesh)))
        np.savetxt(f,mesh,'%f %f %f %d %d %d')
    print("Point cloud was generated and saved!")

###################################################### IMPLEMENTATION ########################################################

###################################### Processing the initial image pair:
print("executing the pipeline with initial image pair...")

initial = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

#Intial projection matrix
proj_1 = np.matmul(K,initial)
trans_12 = np.empty((3,4))

#setting the initial projection as reference
proj_prev = proj_1
result_point_cloud = np.zeros((1,3))
result_point_colours = np.zeros((1,3))

#down-scaling images to make it computationally efficient
image1 = scale_down_image(ImgDataset[0],scale_down_factor)
image2 = scale_down_image(ImgDataset[1],scale_down_factor)

#Extracting features from first set of images
kp1,kp2 = feature_extraction_and_matching(image1,image2)

#computing essential matrix E:
E,mask = cv2.findEssentialMat(kp1,kp2,K,method=cv2.RANSAC,prob=0.999,threshold=0.4,mask=None)
kp1 = kp1[mask.ravel()==1]
kp2 = kp2[mask.ravel()==1]

#obtain the relative translation and rotation with the essential matrix:
_,rot,trans,mask = cv2.recoverPose(E,kp1,kp2,K) 
trans = trans.ravel()

#generate rotation and translation further:
trans_12[:3,:3]=np.matmul(rot,initial[:3,:3])
trans_12[:3,3]=initial[:3, 3]+np.matmul(initial[:3, :3],trans)
proj_2=np.matmul(K,trans_12)
kp1=kp1[mask.ravel()>0]
kp2=kp2[mask.ravel()>0]

#Triangulation for obtaining a reference point cloud:
kp1,kp2,cloud = triangulate(proj_1,proj_2,kp1,kp2)
cloud = cv2.convertPointsFromHomogeneous(cloud.T)

#Calculating the reprojection error and using the pnp method:
error,cloud,repro_pts=get_reproj_err(cloud,kp2.T,trans_12,K)
Rot,trans,kp2,cloud,kp1t=p_n_p(cloud[:,0,:],kp1.T,kp2.T,K)


################################################### Iterating over remaining images in dataset:
for i in range(len(ImgDataset)-2):
    
    if i>0:
        #Setting up the reference keypoints:
        kp1, kp2, cloud = triangulate(proj_1,proj_2,kp1,kp2)
        kp2 = kp2.T

        cloud = cv2.convertPointsFromHomogeneous(cloud.T)
        cloud = cloud[:,0,:]
    
    #Acquiring new image and finding a set of good matches
    new_image = ImgDataset[i+2]
    new_image = scale_down_image(new_image,scale_down_factor)
    kp2_dash, kp3 = feature_extraction_and_matching(image2,new_image)
    
    #Finding common keypoints present in all the 3 images:
    index, kp2_dash_common, kp3_common, kp2_dash_uncommon, kp3_uncommon = trifocal_view(kp2,kp2_dash,kp3)
    
    #putting these common points in the pnp pipeline:
    rot, trans, kp3_common, cloud, kp2_dash_common = p_n_p(cloud[index],kp3_common,kp2_dash_common,K)
    
    #Calculating the new projection matrix
    trans_mat_new = np.hstack((rot,trans))
    proj_new = np.matmul(K,trans_mat_new)

    #calculating the reprojection error from common points in img(i+1)
    error1,cloud,kp_projected=get_reproj_err(cloud,kp3_common,trans_mat_new,K)
   
    #Updating the 3D point cloud:
    kp2_dash_uncommon, kp3_uncommon, cloud = triangulate(proj_2,proj_new,kp2_dash_uncommon,kp3_uncommon)
    cloud = cv2.convertPointsFromHomogeneous(cloud.T)
    error2, cloud, kp_reprojected = get_reproj_err(cloud,kp3_uncommon.T,trans_mat_new,K)
    
    #cummulative error in reprojection: 
    print("Reprojection Error after "+str(i+3)+" images:"+str(np.round(error1+error2,4))) 
    
    #final the point cloud stacked
    result_point_cloud = np.vstack((result_point_cloud,cloud[:,0,:]))
    
    #Finding and stacking the colours of the 3D points in point cloud:
    kp_for_intensity = np.array(kp3_uncommon, dtype=np.int32)
    colors=np.array([new_image[intensity[1], intensity[0]] for intensity in kp_for_intensity.T])
    result_point_colours=np.vstack((result_point_colours,colors)) 
    
    #updating values for next iteration
    proj_1=proj_2
    proj_2=proj_new
    image2=new_image
    kp1=kp2_dash
    kp2=kp3

output(result_point_cloud,result_point_colours)



