import TFL_man 
import json
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_playlist(play_list_path):
    with open(play_list_path) as j:
        playlist = json.load(j)
   
    with open(playlist["pkl"], 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    focal = data['flx']
    pp = data['principle_point']
    
    for i in range(len(playlist["frames"]) - 1):
        prev_frame_id = int(playlist["frames"][i][78:80])
        curr_frame_id = int(playlist["frames"][i + 1][78:80])
        EM = np.eye(4)
        for j in range(prev_frame_id, curr_frame_id):
            EM = np.dot(data['egomotion_' + str(j) + '-' + str(j + 1)], EM)
        tfl_man = TFL_man.TFLMan(EM, pp, focal, playlist["frames"][i], playlist["frames"][i + 1])
        tfl_man.run()

load_playlist("./playlist/playlist_1.json")
    