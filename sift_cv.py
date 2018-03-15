import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt

class SiftFeature():
    def __init__(self,SIFT_INIT_SIGMA = 0.5,SIFT_MAX_INTERP_STEPS = 5,contr_thr = 0.04,r  = 10):
        self.SIFT_INIT_SIGMA = SIFT_INIT_SIGMA
        self.SIFT_MAX_INTERP_STEPS = SIFT_MAX_INTERP_STEPS
        self.contr_thr = contr_thr
        self.r = r

    def interp_hist_peak(self,l,c,r):
        if abs(l-2.0*c+r)<10**(-6):
            return 0
        return 0.5*(l-r)/(l-2.0*c+r)

    def add_good_ori_feature(self,smooth_ori_hist,mag_thr,num_bins = 36):
        hist = smooth_ori_hist
        new_roi_hist = np.zeros(shape=[num_bins])
        for i in  range(num_bins):
            l = (i+num_bins-1)%num_bins
            r = (i+num_bins+1)%num_bins
            c = i
            if hist[c]>hist[l] and hist[c]>hist[r] and hist[c]>mag_thr:
                bin = c + self.interp_hist_peak(l = l,c =c ,r = r)
                if bin<0:
                    bin = (abs(num_bins+bin))%num_bins
                elif bin>=num_bins:
                    bin = (bin-num_bins)%num_bins
                else:
                    bin = bin

                bin = np.round(bin)
                bin = bin.astype(np.int)
                new_roi_hist[bin] = hist[c]+self.interp_hist_peak(l = hist[l],c = hist[c],r = hist[r])
        return new_roi_hist

    def cal_smooth_ori_hist(self,ori_hist):
        num_bins = len(ori_hist)
        smooth_ori_hist = np.zeros(shape=[num_bins])
        for i in range(num_bins):
            smooth_ori_hist[i] = 0.25*ori_hist[(i+num_bins-1)%num_bins]+\
                                 0.5*ori_hist[i]+\
                                 0.25*ori_hist[(i+num_bins+1)%num_bins]
        return smooth_ori_hist

    def cal_ori_hist(self,gauss_pyr,keypoint,sigma,num_bins = 36):
        rad = np.round(3*1.5*sigma)
        rad = rad.astype(np.int)
        o = keypoint[1]
        s = keypoint[2]
        r = keypoint[3]
        c = keypoint[4]
        L = gauss_pyr[o][s,:]
        im_width = L.shape[1]
        im_height = L.shape[0]
        hist = np.zeros(shape=[num_bins])
        exp_denom = 2.0*sigma*sigma
        for i in range(-1*rad,rad+1):
            for j in range(-1*rad,rad+1):
                r_index = r+i
                c_index = c+j
                if r_index<=0 or c_index<=0 or r_index>=im_height-1 or c_index>=im_width-1:
                    continue
                mag,theta = self.cal_mag_ori(r_index,c_index,L)
                weight = np.exp(-(i*i+j*j)/exp_denom)
                bin = np.round(num_bins*(theta+np.pi)/(np.pi*2))
                bin = bin.astype(np.int)
                if bin>=num_bins:
                    bin = 0
                hist[bin] = hist[bin]+weight*mag
        return hist

    def cal_feature_oris(self,gauss_pyr,keypoints,num_bins = 36):
        keypts_sigma = sift_featrue.keypoints_sigma(gauss_pyr=gauss_pyr, keypoints=keypoints)
        feature_oris = []
        for i in range(len(keypoints)):
            keypt = keypoints[i]
            sigma = keypts_sigma[i]
            ori_hist = self.cal_ori_hist(gauss_pyr=gauss_pyr,
                                         keypoint=keypt,
                                         sigma=sigma,
                                         num_bins=num_bins)
            smooth_ori_hist = self.cal_smooth_ori_hist(ori_hist=ori_hist)
            max_mag = np.max(smooth_ori_hist)
            new_ori_hist = self.add_good_ori_feature(smooth_ori_hist = smooth_ori_hist,
                                                     mag_thr=max_mag*0.8)
            feature_oris.append(new_ori_hist)
        return np.array(feature_oris)

    def cal_mag_ori(self,i,j,L):
        mag = np.sqrt(np.square(L[i+1,j]-L[i-1,j])+\
                  np.square(L[i,j+1]-L[i,j-1]))
        theta = np.arctan((L[i,j+1]-L[i,j-1])/(L[i+1,j]-L[i-1,j]))
        return mag,theta

    def keypoints_sigma(self,gauss_pyr,keypoints,sigma = 0.5):
        keypts_sigma = []
        s_num = gauss_pyr[0].shape[0]-3
        for keypt in keypoints:
            o = keypt[1]
            s = keypt[2]
            keypt_sigma = sigma*pow(2,o+s/s_num)
            keypts_sigma.append(keypt_sigma)
        return keypts_sigma

    def Hessian_2D(self,dog,i,j):
        point = dog[i, j]
        dxx = (dog[i, j + 1] + dog[i, j - 1] - 2 * point)
        dyy = (dog[i + 1, j] + dog[i - 1, j] - 2 * point)
        dxy = (dog[i + 1, j + 1] - dog[i + 1, j - 1] -
               dog[i - 1, j + 1] + dog[i - 1, j - 1]) / 4.0

        return np.array([[dxx, dxy],
                         [dxy, dyy]])

    def edge_line(self,Dog_pyr,o,s,i,j):
        dog = Dog_pyr[o][s,:]
        dD_2D = self.Hessian_2D(dog = dog,i = i,j = j)
        dxx = dD_2D[0,0]
        dxy = dD_2D[0,1]
        dyy = dD_2D[1,1]
        tr = dxx + dyy
        det = dxx*dyy - dxy*dxy
        if det<=0:
            return True
        r = self.r
        if tr*tr/det<(r+1.0)*(r+1.0)/r:
            return False
        return True

    def interpolation_extremum(self,Dog_pyr,o,s,i,j):
        point = Dog_pyr[o][s,i,j]
        s_num = Dog_pyr[o].shape[0]
        i_num = Dog_pyr[o].shape[1]
        j_num = Dog_pyr[o].shape[2]

        for step in range(self.SIFT_MAX_INTERP_STEPS):
            if s<=0 or i<=0 or j<=0 or s>=s_num-1 or i>=i_num-1 or j>=j_num-1:
                return None
            X = self.offset(Dog_pyr = Dog_pyr, o = o, s = s, i = i, j = j)
            xi = X[0];xj = X[1];xs = X[2];
            if abs(xi)<0.5 and abs(xj)<0.5 and abs(xs)<0.5:
                break
            s = int(np.round(xs) + s)
            i = int(np.round(xi) + i)
            j = int(np.round(xj) + j)
            if s<0 or i<0 or j<0 or s>=s_num or i>=i_num or j>=j_num:
                return None
        if step>=self.SIFT_MAX_INTERP_STEPS-1:
            return None
        dD = self.deriv_3D(Dog_pyr,o = o, s = s,i = i,j = j)
        D = point+0.5*np.dot(dD,X)
        if D<self.contr_thr/(s_num-2):
            return None
        return [D,o,s,i,j]

    def offset(self,Dog_pyr,o,s,i,j):
        dD = self.deriv_3D(Dog_pyr=Dog_pyr,o=o,s = s, i = i, j = j)
        H = self.hessian_3D(Dog_pyr=Dog_pyr,o = o,s = s, i = i, j = j)
        H_inv = cv2.invert(src=H,flags=cv2.DECOMP_SVD)[1]
        X = np.dot(-1*H_inv,dD)
        return X

    def hessian_3D(self,Dog_pyr,o,s,i,j):
        dogs = Dog_pyr[o]
        point = dogs[s,i,j]
        dxx = (dogs[s,i,j+1]+dogs[s,i,j-1]-2*point)
        dyy = (dogs[s,i+1,j]+dogs[s,i-1,j]-2*point)
        dss = (dogs[s+1,i,j]+dogs[s-1,i,j]-2*point)
        dxy = (dogs[s,i+1,j+1]-dogs[s,i+1,j-1]-
               dogs[s,i-1,j+1]+dogs[s,i-1,j-1])/4.0
        dxs = (dogs[s+1,i,j+1]-dogs[s+1,i,j-1]-
               dogs[s-1,i,j+1]+dogs[s-1,i,j-1])/4.0
        dys = (dogs[s+1,i+1,j]-dogs[s+1,i-1,j]-
               dogs[s-1,i+1,j]+dogs[s-1,i-1,j])/4.0
        return np.array([[dxx,dxy,dxs],
                         [dxy,dyy,dys],
                         [dxs,dys,dss]])

    def deriv_3D(self,Dog_pyr,o,s,i,j):
        dogs = Dog_pyr[o]
        point = dogs[s,i,j]
        dx = dogs[s,i,j+1] - point
        dy = dogs[s,i+1,j] - point
        ds = dogs[s+1,i,j] - point
        return np.array([dx,dy,ds])

    def extreme_point(self,Dog_pyr,o,s,i,j):
        dogs = Dog_pyr[o]
        point = dogs[s,i,j]
        for s_index in range(s-1,s+2):
            for i_index in range(i-1,i+2):
                for j_index in range(j-1,j+2):
                    if s_index == s and i_index == i and j_index == j:
                        continue
                    if point < 0 and point > dogs[s_index,i_index,j_index]:
                        return False
                    if point >= 0 and point < dogs[s_index,i_index,j_index]:
                        return False
        return True

    def key_point_search(self,Dog_pyr):
        keypoints = []
        for o in range(len(Dog_pyr)):
            for s in range(1,Dog_pyr[o].shape[0]-1):
                for i in range(1,Dog_pyr[o][s].shape[0]-1):
                    for j in range(1,Dog_pyr[o][s].shape[1]-1):
                        if abs(Dog_pyr[o][s,i,j])<0.03:
                            continue
                        if self.extreme_point(Dog_pyr=Dog_pyr,o = o,s = s,i = i,j = j):
                            keypoint = self.interpolation_extremum(Dog_pyr = Dog_pyr,o = o, s = s,i = i, j = j)
                            if keypoint is not None:
                                if self.edge_line(Dog_pyr = Dog_pyr,o = keypoint[1],s = keypoint[2],
                                                  i = keypoint[3],j = keypoint[4]) == False:
                                    keypoints.append(keypoint)
        return keypoints

    def create_dog_pyr(self,gauss_pyr):
        Dog_pyr = []
        for octaves in gauss_pyr:
            Dogs = []
            for i in range(1,octaves.shape[0]):
                dog = octaves[i] - octaves[i-1]
                Dogs.append(dog)
            Dogs = np.array(Dogs)
            Dog_pyr.append(Dogs)
        return Dog_pyr

    def create_sigmas(self,octaves,S,sigma):
        storeies = S+3
        sigmas = np.zeros(shape=[octaves, storeies])
        k = pow(2, 1 / S)
        for i in range(octaves):
            sigma_i_origin = sigma * pow(2, i)
            for j in range(storeies):
                sigmas[i, j] = sigma_i_origin * pow(k, j)
        return sigmas

    def create_guass_pyr(self,init_img,octaves,S = 3,sigma = None):
        storeies = S+3
        # origin_img = init_img[0:init_img.shape[0]:2,0:init_img.shape[1]:2]
        # cv2.imshow('img',img)
        if sigma is None:
            sigma = self.SIFT_INIT_SIGMA
        guass_pyr = []

        sigmas = self.create_sigmas(octaves,S,sigma)
        # print(sigmas)
        for i in range(octaves):
            imgs = []
            origin_img = init_img[0:init_img.shape[0]:int(pow(2,i+1)),
                         0:init_img.shape[1]:int(pow(2,i+1))]
            for j in range(storeies):
                if i == 0 and j == 0:
                    imgs.append(origin_img)
                    # cv2.imshow('origin',origin_img)
                    continue
                else:
                    sig = sigmas[i,j]
                    size = int(sig*6+1)
                    if size%2 == 0:
                        size = size+1
                    img = cv2.GaussianBlur(src=origin_img,ksize=(size,size),sigmaX=sig,sigmaY=sig)
                    imgs.append(img)
                    # cv2.imshow('img'+str(i)+str(j),img)
            imgs = np.array(imgs)
            # imgs = imgs.astype(np.float32)
            guass_pyr.append(np.array(imgs))
        return guass_pyr

    def create_minus_one_img(self,img,sigma = 1.6):
        SIFT_INIT_SIGMA = self.SIFT_INIT_SIGMA
        sig = np.sqrt(sigma*sigma-SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4)
        init_img = cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
        init_img = cv2.resize(src=init_img,
                              dsize=(img.shape[1]*2,img.shape[0]*2),
                              fx=2,
                              fy=2,
                              interpolation=cv2.INTER_CUBIC)
        init_img = cv2.GaussianBlur(src=init_img,ksize=(0,0),sigmaX=sig,sigmaY=sig)
        return init_img

    def draw_sift_features(self,image,keypoints,feature_rois):
        for i in range(len(keypoints)):
            keypt = keypoints[i]
            feature_roi = feature_rois[i]
            o = keypt[1]
            x1 = keypt[3]*pow(2,o)
            y1 = keypt[4]*pow(2,o)
            x1 = int(x1)
            y1 = int(y1)
            if x1>=image.shape[1] or y1>=image.shape[0]:
                continue
            cv2.circle(image,center=(y1,x1),radius=2,color=(0,255,0))
            theta_per = np.pi/len(feature_roi)
            for k in range(len(feature_roi)):
                mag = feature_roi[k]
                if mag>10**(-6):
                    x2 = np.round(x1+mag*np.cos(k*theta_per+theta_per/2)).astype(np.int)
                    y2 = np.round(y1+mag*np.sin(k*theta_per+theta_per/2)).astype(np.int)
                    cv2.arrowedLine(img=image,pt1=(y1,x1),pt2=(y2,x2),color=(255,0,0))
        cv2.imshow('sift',image)
        cv2.waitKey()
        return image

if __name__ == '__main__':
    image = cv2.imread('lena.jpg')
    image = image.astype(np.float32)
    sift_featrue = SiftFeature()
    init_img = sift_featrue.create_minus_one_img(image)

    gauss_pyr = sift_featrue.create_guass_pyr(init_img=init_img,octaves=4)
    dogs_pyr = sift_featrue.create_dog_pyr(gauss_pyr=gauss_pyr)
    keypoints = sift_featrue.key_point_search(Dog_pyr=dogs_pyr)
    print(len(keypoints))
    print(type(keypoints))
    # keypts_sigma = sift_featrue.keypoints_sigma(gauss_pyr=gauss_pyr,keypoints=keypoints)
    # print(len(keypts_sigma))
    # print(keypts_sigma)
    feature_rois = sift_featrue.cal_feature_oris(gauss_pyr=gauss_pyr,keypoints=keypoints)
    print(feature_rois.shape)
    sift_image = sift_featrue.draw_sift_features(image=image.astype(np.uint8),
                                                 keypoints=keypoints,
                                                 feature_rois=feature_rois)
    cv2.imwrite('sift_lena.jpg',sift_image)
    # print(feature_rois)
    # print(keypoints)
    # for i in range(len(gauss_pyr)):
    #     print(gauss_pyr[i].shape)
    #     for j in range(gauss_pyr[i].shape[0]):
    #         index = i*gauss_pyr[i].shape[0]+j+1
    #         print('index = ',index)
    #         plt.subplot(len(gauss_pyr), gauss_pyr[i].shape[0], index)
    #         plt.imshow(gauss_pyr[i][j],plt.cm.gray)
    #         plt.axis('off')
    # plt.show()
    # for i in range(len(dogs_pyr)):
    #     print(dogs_pyr[i].shape)
    #     for j in range(dogs_pyr[i].shape[0]):
    #         index = i*dogs_pyr[i].shape[0]+j+1
    #         print('index = ',index)
    #         plt.subplot(len(dogs_pyr), dogs_pyr[i].shape[0], index)
    #         plt.imshow(dogs_pyr[i][j],plt.cm.gray)
    #         plt.axis('off')
    # plt.show()
    # cv2.imshow('origin',image)
    # cv2.imshow('lena',init_img)
    # cv2.waitKey()