from multiprocessing import Pool
import LasFilePreprocessing as LFP
import open3d as o3d
import pptk


import time
import os


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def task(x):
    return x*2

TileDivision = 60 #test purposes
#store Get_subtileArray() result
rows, cols = (TileDivision, TileDivision)
Matrix_Buffer = [[0]*cols]*rows

def CreateSubtileArr_P(TileObj,row_ID,col_ID):
    info('Funtion - CreateSubtileArr_P()')
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

    #lasTile class
    TileObj = LFP.lasTile(LiDAR_Dataframe=SR_df,TileDivision=5)

    #Serialized
    s_start = time.time()
    lidar_TilesubsetArr = TileObj.Get_subtileArray()
    #sanity check
    #print(lidar_TilesubsetArr[0][0].shape)

    s_end = time.time()
    stime = s_end - s_start
    print("Serial Time : ",stime)

    test_start = time.time()
    a = [[[i,j]for j in range(TileDivision)] for i in range(TileDivision)] #NOTE: Bottleneck here, as TileDivision increases(X6), list comprehension is approximately 30times slower
    test_end = time.time()
    print("List Comp time test = ", test_end - test_start)

    #Parrallelized
    p_start = time.time()
    X_div_len, Y_div_len = TileObj.Get_SubTileDimensions()

    with Pool(5) as p:
        [[p.apply_async(CreateSubtileArr_P, args=(TileObj, i, j)) for j in range(TileDivision)] for i in range(TileDivision)]
    p_end = time.time()
    ptime = p_end - p_start
    print("Parallel Time : ",ptime)

    print("% increase = ",(stime/ptime)*100)

