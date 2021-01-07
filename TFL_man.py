from phase_1.run_attention import find_tfl_lights
import phase_3.SFM
from phase_3.SFM import calc_TFL_dist, unnormalize, rotate, prepare_3D_data
from tensorflow.keras.models import load_model 
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

class suspicious_points:
    def __init__(self):
        self.red_x = list()
        self.red_y = list()
        self.green_x = list()
        self.green_y = list()


    def set_values(self,red_x, red_y, green_x, green_y):
        self.red_x = red_x
        self.red_y = red_y
        self.green_x = green_x
        self.green_y = green_y

class FrameContainer(object):
    def __init__(self, img_path):
        self.img = Image.open(img_path) 
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind=[]
        self.valid=[]

class TFLMan:
    def __init__(self, EM_matrix, Principal_point, Focal_length, Frame_prev, Frame_curr):
        self.EM_matrix = EM_matrix
        self.Principal_point = Principal_point
        self.Focal_length = Focal_length
        self.Frame_prev = Frame_prev
        self.Frame_curr = Frame_curr
        self.suspicious_points_prev = suspicious_points()
        self.suspicious_points_curr = suspicious_points()


    def run_phase_1(self, frame, suspicious_points):
        red_x, red_y, green_x, green_y = find_tfl_lights(frame)
        suspicious_points.set_values(red_x, red_y, green_x, green_y)


    def run_phase_2(self,img_path, suspicious_points):
        model = load_model("./phase_2/model.h5")
        predicted_red_x = list()
        predicted_red_y = list()
        # filter the red points
        for i in range(len(suspicious_points.red_x)):
            crop_image = Image.open(img_path)
            crop_image = np.array(crop_image.crop((suspicious_points.red_x[i] - 40, suspicious_points.red_y[i] - 40, suspicious_points.red_x[i] + 41, suspicious_points.red_y[i] + 41)))
            res = model.predict(np.array([crop_image]))
            if res[0][1] > 0.5:
                predicted_red_x.append(suspicious_points.red_x[i])
                predicted_red_y.append(suspicious_points.red_y[i])

        predicted_green_x = list()
        predicted_green_y = list()
        # filter the green points
        for i in range(len(suspicious_points.green_x)):
            crop_image = Image.open(img_path)
            crop_image = np.array(crop_image.crop((suspicious_points.green_x[i] - 40, suspicious_points.green_y[i] - 40, suspicious_points.green_x[i] + 41, suspicious_points.green_y[i] + 41)))
            res = model.predict(np.array([crop_image]))
            if res[0][1] > 0.5:
                predicted_green_x.append(suspicious_points.green_x[i])
                predicted_green_y.append(suspicious_points.green_y[i])

        suspicious_points.set_values(predicted_red_x, predicted_red_y, predicted_green_x, predicted_green_y)
    

    def run_phase_3(self):
        prev_container = FrameContainer(self.Frame_prev)
        curr_container = FrameContainer(self.Frame_curr)

        prev_points = list()
        for i in range(len(self.suspicious_points_prev.red_x)):
            prev_points.append([self.suspicious_points_prev.red_x[i], self.suspicious_points_prev.red_y[i]])
        for i in range(len(self.suspicious_points_prev.green_x)):
            prev_points.append([self.suspicious_points_prev.green_x[i], self.suspicious_points_prev.green_y[i]])

        prev_container.traffic_light = np.array(prev_points)

        curr_points = list()
        for i in range(len(self.suspicious_points_curr.red_x)):
            curr_points.append([self.suspicious_points_curr.red_x[i], self.suspicious_points_curr.red_y[i]])
        for i in range(len(self.suspicious_points_curr.green_x)):
            curr_points.append([self.suspicious_points_curr.green_x[i], self.suspicious_points_curr.green_y[i]])
            
        curr_container.traffic_light = np.array(curr_points)

        curr_container.EM = self.EM_matrix

        curr_container = calc_TFL_dist(prev_container, curr_container, self.Focal_length, self.Principal_point)
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(prev_container, curr_container, self.Focal_length, self.Principal_point)
        norm_rot_pts = rotate(norm_prev_pts, R)
        rot_pts = unnormalize(norm_rot_pts, self.Focal_length, self.Principal_point)
        foe = np.squeeze(unnormalize(np.array([norm_foe]), self.Focal_length, self.Principal_point))
        return curr_container, foe, rot_pts

    def run(self):

        id_frame_curr = self.Frame_curr[78:80]
        self.run_phase_1(self.Frame_prev, self.suspicious_points_prev)
        fig, (phase_1_curr, phase_2_curr,phase_3_curr) = plt.subplots(3, 1)
        fig.suptitle("Frame #" + id_frame_curr)

        self.run_phase_1(self.Frame_curr, self.suspicious_points_curr)
        phase_1_curr.set_ylabel('candidates')
        phase_1_curr.plot(self.suspicious_points_curr.red_x, self.suspicious_points_curr.red_y, 'r+', markersize=4)
        phase_1_curr.plot(self.suspicious_points_curr.green_x, self.suspicious_points_curr.green_y, 'g+', markersize=4)
        phase_1_curr.imshow(np.array(Image.open(self.Frame_curr)))
        self.run_phase_2(self.Frame_curr, self.suspicious_points_curr)
        phase_2_curr.set_ylabel("traffic lights")
        phase_2_curr.plot(self.suspicious_points_curr.red_x, self.suspicious_points_curr.red_y, 'r+', markersize=4)
        phase_2_curr.plot(self.suspicious_points_curr.green_x, self.suspicious_points_curr.green_y, 'g+', markersize=4)
        phase_2_curr.imshow(np.array(Image.open(self.Frame_curr)))
        
        curr_container, foe , rot_pts = self.run_phase_3()


        phase_3_curr.set_ylabel("distance")
        phase_3_curr.imshow(curr_container.img)
        curr_p = curr_container.traffic_light
        phase_3_curr.plot(curr_p[:,0], curr_p[:,1], 'b+')

        for i in range(len(curr_p)):
            if curr_container.valid[i]:
                phase_3_curr.text(curr_p[i,0], curr_p[i,1], r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
        phase_3_curr.plot(foe[0], foe[1], 'r+')

        plt.show()
        
    