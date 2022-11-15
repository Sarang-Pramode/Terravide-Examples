from multiprocessing import Pool
import LasFilePreprocessing as LFP
import ExtractGroundPlane as GP
import pptk
import progressbar

import time
import os

def View3Dpoints(points):
    """Calls PPTK with basic config to plot 3d points

    Args:
        points (Nx3 Numpy Array): NX3 numpy array
    """
    exitViewerFlag = False
    while not exitViewerFlag:
        v = pptk.viewer(points)
        exitViewerFlag = int(input("Enter a 1 to exit viewer : "))

    v.close()

    return None

# #Parrallize creation of Matrix Buffer
# def CreateSubtileArr_P(TileObj,row_ID,col_ID):
#     X_div_len, Y_div_len = TileObj.Get_SubTileDimensions()
#     Matrix_Buffer[row_ID][col_ID] = TileObj.Get_subtile(X_div_len, Y_div_len, row_ID, col_ID)

# #Parrallelized
# p_start = time.time()
# X_div_len, Y_div_len = TileObj.Get_SubTileDimensions()

# with Pool(5) as p:
#     [[p.apply_async(CreateSubtileArr_P, args=(TileObj, i, j)) for j in range(TileDivision)] for i in range(TileDivision)]
# p_end = time.time()
# ptime = p_end - p_start
# print("Parallel Time : ",ptime)


if __name__ == '__main__':

    print("Ground Plane Classifcation Algorithm")

    TileDivision = 60 #test purposes
    rows, cols = (TileDivision, TileDivision)

    lasfilepath = 'Datasets/FTP_files/LiDAR/NYC_2017/25192.las'
    lasfile_object = LFP.Read_lasFile(lasfilepath)
    lidar_df, rawpoints = LFP.Create_lasFileDataframe(lasfileObject=lasfile_object)

    MR_df = LFP.Get_MRpoints(lidar_df)
    SR_df = LFP.Get_SRpoints(lidar_df)

    #lasTile class
    TileObj = LFP.lasTile(SR_df,TileDivision)

    #Serialized
    s_start = time.time()
    lidar_TilesubsetArr = TileObj.Get_subtileArray()

    s_end = time.time()
    stime = s_end - s_start
    print("Extraction of Subtile Matrix Buffer Serial Time : ",stime)

    #Ground Plane Classifcation - Serial Implementation
    g_start = time.time()
    Potential_Ground_Points = []
    Other_points = []

    GP_obj = GP.GP_class()

    for row in range(TileDivision):
        for col in range(TileDivision):

            tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()

            Ground_Points, Not_ground_points = GP_obj.Extract_GroundPoints(tile_segment_points)

            for k in Ground_Points:
                Potential_Ground_Points.append(k) #append points which may be potentially ground points
            for l in Not_ground_points:
                Other_points.append(l)
    
    g_end = time.time()
    gtime = g_end - g_start
    print("Ground Point Extraction Algorithm Serial Time : ",gtime)

    exitViewerFlag = False
    while not exitViewerFlag:
        v = pptk.viewer(Potential_Ground_Points)
        exitViewerFlag = int(input("Enter a 1 to exit viewer"))

    v.close()

