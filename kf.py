from cProfile import label
import math
import numpy as np
from numpy.linalg import multi_dot
from numpy.linalg import inv
import re
import matplotlib.pyplot as plt

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

def kalmanFilter(s_i, index):
    X_priori = multi_dot([s_i["A"], s[index - 1]["X"]])

    P_priori = np.add(multi_dot([s_i["A"], s[index - 1]["P"], np.transpose(s_i["A"])]), Q)

    K = multi_dot([P_priori, np.transpose(H), inv(np.add(multi_dot([H, P_priori, np.transpose(H)]), s_i["R"]))])

    X = np.add(X_priori, multi_dot([K, np.subtract(s_i["Z"], multi_dot([H, X_priori]))]))

    P = np.subtract(P_priori, multi_dot([K, H, P_priori]))

    return {"X": X, "P": P}

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
delta_t = 0.001
total = len(odom_x)

H = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

Q = np.array([
    [0.00001, 0, 0, 0, 0],
    [0, 0.00001, 0, 0, 0],
    [0, 0, 0.001, 0, 0],
    [0, 0, 0, 0.001, 0],
    [0, 0, 0, 0, 0.001]
])

s = [{}] * total

s[0]["P"] = np.array([
    [0.01, 0, 0, 0, 0],
    [0, 0.01, 0, 0, 0],
    [0, 0, 0.01, 0, 0],
    [0, 0, 0, 0.01, 0],
    [0, 0, 0, 0, 0.01]
])

omega = V * math.tan(odom_theta[0]) / L

s[0]["X"] = np.array([odom_x[0], odom_y[0], V, odom_theta[0], omega])

for i in range(total - 1):
    A = np.array([
        [1, 0, delta_t * math.cos(odom_theta[i]), 0, 0],
        [0, 1, delta_t * math.sin(odom_theta[i]), 0, 0],
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

    Z = np.array([gps_x[i], gps_y[i], V, imu_heading[i], omega])

    s[i]["A"] = A
    s[i]["R"] = R
    s[i]["Z"] = Z

    if i > 0:
        s[i + 1] = kalmanFilter(s[i], i)

x, y = [], []
for i in range(len(s)):
    x.append(s[i]['X'][0])
    y.append(s[i]['X'][1])

plt.plot(x, y, label="kalman")
plt.plot(odom_x, odom_y, label="odm")
plt.plot(gps_x, gps_y, label="gps")
plt.legend()
plt.savefig("graph.png")