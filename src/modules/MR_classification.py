import laspy
import numpy as np
import pandas as pd


def dummy_func():
    print("Success")

    return 0

#Read las file

las_filename = "25192"
las_file_year = 2015
las_file_path = "/Users/sarangpramode/Desktop/Hub/treefolio/streetTrees/Testbed/lidar_files/"+las_filename+".las"
las_testfile = laspy.read(las_file_path) # .las file taken from NYC topbathymetric 2017 Lidar data

#Making a datframe from the lidar data

Xscale = las_testfile.header.x_scale
Yscale = las_testfile.header.y_scale
Zscale = las_testfile.header.z_scale

Xoffset = las_testfile.header.x_offset
Yoffset = las_testfile.header.y_offset
Zoffset = las_testfile.header.z_offset

lidarPoints = np.array(
    ( (las_testfile.X*Xscale)/3.28 + Xoffset,  # convert ft to m and correct measurement
      (las_testfile.Y*Yscale)/3.28 + Yoffset,
      (las_testfile.Z*Zscale)/3.28 + Zoffset,
    las_testfile.intensity,
    las_testfile.classification,
    las_testfile.return_number, 
    las_testfile.number_of_returns)).transpose()
lidar_df = pd.DataFrame(lidarPoints , columns=['X','Y','Z','intensity','classification','return_number','number_of_returns'])

# Removing groung plane classified points
#lidar_df = lidar_df[lidar_df.classification != 2]

#Raw point cloud data
points = np.array((
    ((lidar_df.X)*(Xscale)) + Xoffset, # convert ft to m
    (lidar_df.Y)*(Yscale) + Yoffset, #convert ft to m
    (lidar_df.Z)*(Zscale) + Zoffset
)).transpose()

print(points.shape)

