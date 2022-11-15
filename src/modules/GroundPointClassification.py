from multiprocessing import Pool
import LasFilePreprocessing as LFP
import ExtractGroundPlane
import pptk

import time
import os


TileDivision = 10 #test purposes
#store Get_subtileArray() result
rows, cols = (TileDivision, TileDivision)
Matrix_Buffer = [[0]*cols]*rows

def CreateSubtileArr_P(TileObj,row_ID,col_ID):
    X_div_len, Y_div_len = TileObj.Get_SubTileDimensions()
    Matrix_Buffer[row_ID][col_ID] = TileObj.Get_subtile(X_div_len, Y_div_len, row_ID, col_ID)


if __name__ == '__main__':
    # with Pool(5) as p:
    #     print(p.map(task, [1, 2, 3]))

    lasfilepath = 'Datasets/FTP_files/LiDAR/NYC_2017/25192.las'
    lasfile_object = LFP.Read_lasFile(lasfilepath)
    lidar_df, rawpoints = LFP.Create_lasFileDataframe(lasfileObject=lasfile_object)

    MR_df = LFP.Get_MRpoints(lidar_df)
    SR_df = LFP.Get_SRpoints(lidar_df)

    #pptk.viewer(SR_df.iloc[:,:3].to_numpy())

    #lasTile class
    TileObj = LFP.lasTile(LiDAR_Dataframe=SR_df,TileDivision=5)

    #Serialized
    s_start = time.time()
    lidar_TilesubsetArr = TileObj.Get_subtileArray()
    #sanity check
    print(lidar_TilesubsetArr[0][0].shape)

    # s_end = time.time()
    # stime = s_end - s_start
    # print("Serial Time : ",stime)

    # test_start = time.time()
    # a = [[[i,j]for j in range(TileDivision)] for i in range(TileDivision)] #NOTE: Bottleneck here, as TileDivision increases(X6), list comprehension is approximately 30times slower
    # test_end = time.time()
    # print("List Comp time test = ", test_end - test_start)

    # #Parrallelized
    # p_start = time.time()
    # X_div_len, Y_div_len = TileObj.Get_SubTileDimensions()

    # with Pool(5) as p:
    #     [[p.apply_async(CreateSubtileArr_P, args=(TileObj, i, j)) for j in range(TileDivision)] for i in range(TileDivision)]
    # p_end = time.time()
    # ptime = p_end - p_start
    # print("Parallel Time : ",ptime)

    # print("% increase = ",(stime/ptime)*100)

    pptk.viewer(lidar_TilesubsetArr[0][0])

    #Ground Plane Classifcation - Serial Implementation
    g_start = time.time()
    # with Pool(5) as p:
    #     [[p.apply_async(ExtractGroundPlane.Extract_GroundPointsAlgo, args=(TileObj.Matrix_Buffer[i][j])) for j in range(TileDivision)] for i in range(TileDivision)]
    Potential_Ground_Points = []
    Other_points = []

    for row in range(TileDivision):
        for col in range(TileDivision):

            tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()

            Ground_Points, Not_ground_points = ExtractGroundPlane.Extract_GroundPointsAlgo(tile_segment_points)

            for k in Ground_Points:
                Potential_Ground_Points.append(k) #append points which may be potentially ground points
            for l in Not_ground_points:
                Other_points.append(l)
    

    g_end = time.time()
    gtime = g_end - g_start
    print("Ground Point Extraction Algorithm Serial Time : ",gtime)

    pptk.viewer(Potential_Ground_Points)

