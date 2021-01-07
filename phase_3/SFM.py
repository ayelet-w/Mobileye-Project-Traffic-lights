import numpy as np
from scipy.spatial import distance

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container

def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ

def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        if corresponding_p_ind == None and corresponding_p_rot == None:
            break
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec

def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    # 𝑥 ≔ (𝑥 − 𝑝𝑝_𝑥) / 𝑓 =𝑋/𝑍
    res = list()
    for point in pts:
      res.append([point[0] - pp[0], point[1] - pp[1]] / focal)
    return np.array(res)

    
def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    # x ≔ (𝑥 * f + 𝑝𝑝_𝑥) 
    res = list()
    for point in pts:
      res.append([point[0] * focal + pp[0], point[1] * focal + pp[1]])
    return np.array(res)

def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3,:3]
    tZ = EM[2,3]
    #foe = (e_x, e_y) = (p_x/t_z, p_y/t_z)
    foe = np.array([EM[0, 3], EM[1, 3]])/tZ
    return R, foe, tZ

def rotate(pts, R):
    # rotate the points - pts using R
    # ((𝑥 ̃_𝑟, 𝑦 ̃_𝑟, 1)) = 1 / 𝑍_𝑟 * 𝑷_𝒓 = 𝑍_𝑝 / 𝑍_𝑟 * R((𝑥_𝑝, 𝑦_𝑝, 1)) ≔ 𝑍_𝑝 / 𝑍_𝑟 * ((a,𝑏,𝑐))
    res = list()
    for point in pts:
      point_rotate = R.dot(np.array([point[0], point[1], 1]))
      # 𝑥_𝑟 = 𝑎/𝑐
      # 𝑦_𝑟 = 𝑏/𝑐 
      point_rotate = (point_rotate[0], point_rotate[1])/point_rotate[2]
      res.append(point_rotate)
    return np.array(res)

def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    # 𝑦 = 𝑚𝑥 + 𝑛 =((𝑒_𝑦 − 𝑦) / (𝑒_𝑥 − 𝑥)) * 𝑥 + (𝑦*𝑒_𝑥 − 𝑒_𝑦*𝑥) / (𝑒_𝑥 − 𝑥)
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = ((p[1] * foe[0]) - (foe[1] * p[0])) / (foe[0] - p[0])
    # the distance of p from 𝑙 is: 𝑑= abs((𝑚𝑥 + 𝑛 − 𝑦) / sqrt(𝑚^2 + 1))
    denominator = np.sqrt(m * m + 1)
    distance = abs(m * norm_pts_rot[0][0] + n - norm_pts_rot[0][1]) / denominator
    closest_index = 0
    closest_point = norm_pts_rot[0]
    for index, point in enumerate(norm_pts_rot):
      curr_distance = abs(m * point[0] + n - point[1]) / denominator
      if curr_distance < distance:
        distance = curr_distance
        closest_index = index
        closest_point = point

    # if distance.euclidean(closest_point , p) > 75:
    #     return None, None
    return closest_index, closest_point


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z

    # 𝒁 = (𝒕_𝒁 ∙ (𝒆_𝒙 − 𝑥_𝑟)) / (𝑥_𝑐 − 𝑥_𝑟)
    z_x = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])

    # 𝒁 = (𝒕_𝒁 ∙ (𝒆_y − y_𝑟)) / (y_𝑐 − y_𝑟)
    z_y = (tZ * (foe[1] - p_rot [1])) / (p_curr[1] - p_rot[1])

    Z_x_w = abs(p_curr[0] - p_rot[0])
    Z_y_w = abs(p_curr[1] - p_rot[1])

    sum_w = Z_x_w + Z_y_w
    if (Z_x_w + Z_y_w) == 0:
        return 0
    Z_x_w /= sum_w
    Z_y_w /= sum_w
    Z = Z_x_w * z_x + Z_y_w * z_y
    
    return Z

