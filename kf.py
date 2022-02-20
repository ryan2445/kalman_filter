import math
import numpy as np
import re

def readFile():
    file = open("EKF_DATA_circle.txt", "r")
    lines = file.readlines()
    file.close()

    header = re.sub("%|\n", "", lines[0]).split(",")
    data = {}
    for i in range(len(header)):
        data[header[i]] = []

    for i in range(1, len(lines)):
        line = re.sub("\n", "", lines[i]).split(",")
        for j in range(len(line)):
            data[header[j]].append(float(line[j]))

    return data

def kalmanFilter(s_i):


    return s_i

data = readFile()

odom_x = data["field.O_x"]
odom_y = data["field.O_y"]
odom_theta = data["field.O_t"]
gps_x = data["field.G_x"]
gps_y = data["field.G_y"]
gps_co_x = data["field.Co_gps_x"]
gps_co_y = data["field.Co_gps_y"]
imu_heading = data["field.I_t"]
imu_co_heading = data["field.Co_I_t"]
V = 0.44
L = 1
# omega = V * math.tan(odom_theta[0]) / L
delta_t = 0.001
total = len(odom_x)

# x = np.array([odom_x[0], odom_y[0], V, odom_theta[0], omega])
Q = np.array([
    [0.00001, 0, 0, 0, 0],
    [0, 0.00001, 0, 0, 0],
    [0, 0, 0.0001, 0, 0],
    [0, 0, 0, 0.0001, 0],
    [0, 0, 0, 0, 0.0001]
])
H = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])
# R = np.array([
#     [0.1, 0, 0, 0, 0],
#     [0, 0.1, 0, 0, 0],
#     [0, 0, 0.01, 0, 0],
#     [0, 0, 0, 0.01, 0],
#     [0, 0, 0, 0, 0.01]
# ])
B = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])
u = np.array([0, 0, 0, 0, 0])
P = np.array([
    [0.01, 0, 0, 0, 0],
    [0, 0.01, 0, 0, 0],
    [0, 0, 0.01, 0, 0],
    [0, 0, 0, 0.01, 0],
    [0, 0, 0, 0, 0.01]
])

s = [{}] * total

for i in range(total - 1):
    A = np.array([
        [1, 0, delta_t*math.cos(odom_theta[i]), 0, 0],
        [0, 1, delta_t*math.sin(odom_theta[i]), 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, delta_t],
        [0, 0, 0, 0, 1]
    ])

    R = np.array([
        [gps_co_x[i], 0, 0, 0, 0],
        [0, gps_co_y[i], 0, 0, 0],
        [0, 0, 0.01, 0, 0],
        [0, 0, 0, imu_co_heading[i], 0],
        [0, 0, 0, 0, 0.01]
    ])

    omega = V * math.tan(odom_theta[i]) / L

    x = np.array([odom_x[i], odom_y[i], V, odom_theta[i], omega])

    z = np.array([gps_x[i], gps_y[i], V, imu_heading[i], omega])

    s[i]["x"] = x
    s[i]["A"] = A
    s[i]["R"] = R
    s[i]["z"] = z

    s[i + 1] = kalmanFilter(s[i])
