import cv2
import numpy as np
import time

def dense_depth_map(pcl, n, m, grid):
	ng = 2 * grid + 1
	# n is shape[0], m is shape[1]
	mX = np.full((n, m), np.inf)
	mY = np.full((n, m), np.inf)
	mD = np.full((n, m), 0)

	mX[list(np.rint(pcl[:, 1]).astype(int)), list(np.rint(pcl[:, 0]).astype(int))] = pcl[:, 0] - np.rint(pcl[:, 0])
	mY[list(np.rint(pcl[:, 1]).astype(int)), list(np.rint(pcl[:, 0]).astype(int))] = pcl[:, 1] - np.rint(pcl[:, 1])
	mD[list(np.rint(pcl[:, 1]).astype(int)), list(np.rint(pcl[:, 0]).astype(int))] = pcl[:, 2]

	kmX, kmY, kmD = [], [], []
	for i in range(ng):
		kmX_, kmY_, kmD_ = [], [], []
		for j in range(ng):
			kmX_.append(mX[i:n-ng+i, j:m-ng+j] - grid + i)
			kmY_.append(mY[i:n-ng+i, j:m-ng+j] - grid + j)
			kmD_.append(mD[i:n-ng+i, j:m-ng+j])
		kmX.append(kmX_)
		kmY.append(kmY_)
		kmD.append(kmD_)
	
	S = np.zeros(kmX[0][0].shape)
	Y = np.zeros(kmX[0][0].shape)
	for i in range(ng):
		for j in range(ng):
			# print(i, j)
			s = 1/np.sqrt(kmX[i][j]**2 + kmY[i][j]**2)
			Y += s * kmD[i][j]
			S += s
	S[S==0] = 1
	output = np.zeros((n, m))
	output[grid:grid+S.shape[0], grid:grid+S.shape[1]] = Y/S
	return output

# Projection Matrix of camera_2 <- velodyne
P = [[609.6954, -721.4216, -1.2513,   -123.0418],
     [180.3842,  7.6448,   -719.6515, -101.0167],
     [0.9999,    1.2437e-4, 0.0105,   -0.2694]]
P = np.array(P)
velodyne = np.fromfile("./data/velodyne/0000000001.bin", dtype=np.float32).reshape(-1, 4)
velodyne = velodyne[velodyne[:, 0] > 5]
img = cv2.imread("./data/image_2/0000000001.png")

velodyne = P.dot(velodyne.T)
velodyne[0, :] /= velodyne[2, :]
velodyne[1, :] /= velodyne[2, :]
print(img.shape)

in_h_range = np.logical_and(velodyne[0, :] > 0, velodyne[0, :] < img.shape[1] - 0.5)
in_v_range = np.logical_and(velodyne[1, :] > 0, velodyne[1, :] < img.shape[0] - 0.5)
# print(velodyne.T[np.logical_and(in_h_range, in_v_range)].shape)

velodyne = velodyne.T[np.logical_and(in_h_range, in_v_range)]
print(velodyne.shape)

## visualize
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for i in range(velodyne.shape[0]):
	# cv2.circle(img, (np.int32(velodyne[i][0]),np.int32(velodyne[i][1])),2, (int(velodyne[i][2]),255,255),-1)
	img[np.int32(velodyne[i][1]), np.int32(velodyne[i][0]), :] = np.array([int(velodyne[i][2]),255,255])
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
cv2.imshow("sparse_depth_image", img)
# cv2.waitKey(0)


start_time = time.time()
output = dense_depth_map(velodyne, img.shape[0], img.shape[1], 3)
print("----- %s seconds -----" % (time.time() - start_time) )

output = 255 * (output - output.min())/(output.max() - output.min())
output = output.astype(np.uint8)
a = np.full(output.shape, 255).astype(np.uint8)
b = np.full(output.shape, 255).astype(np.uint8)
depth = np.stack((a, b, output), axis = 2)
cv2.imshow("depth_depth_image", output)
cv2.waitKey(0)

