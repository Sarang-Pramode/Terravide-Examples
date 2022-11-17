import src.modules.MultipleReturnsClassification as MRC
import src.modules.utils as util
import time
import pptk

# Aliases for simplicity - Preprocessing 
LasHandling = MRC.LFP
# LasProcess = LasHandling.lasTile
#MR_Class = MRC.MR_class()


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



if __name__ == "__main__":

    script_start = time.time()

    print("Multiple Returns Classifcation Algorithm")

    TileDivision = 12
    rows, cols = (TileDivision, TileDivision)

    las_fileID = 25192
    lasfilepath = 'Datasets/FTP_files/LiDAR/NYC_2017/'+str(las_fileID)+'.las'
    
    #Read las file
    lasfile_object = LasHandling.Read_lasFile(lasfilepath)

    #Create Dataframe from lasfile
    lidar_df, rawpoints = LasHandling.Create_lasFileDataframe(lasfileObject=lasfile_object)

    #Extract MR and SR points from Dataframe
    MR_df = LasHandling.Get_MRpoints(lidar_df)
    SR_df = LasHandling.Get_SRpoints(lidar_df)

    #lasTile class
    TileObj_SR = MRC.MR_class(SR_df,TileDivision) #Single Return Points
    TileObj_MR = MRC.MR_class(MR_df,TileDivision) #Multiple Return Points

    #Serialized Creation of Lidar Subtiles
    lidar_TilesubsetArr = TileObj_MR.Get_subtileArray()

    #sanity check
    #temp_segmentPoints = lidar_TilesubsetArr[1][0].iloc[:,:3].to_numpy()

    #Counter : Iterating through each tile
    Tilecounter = 0

    start = time.time()
    Trees_Buffer = []
    for row in range(TileDivision):
        for col in range(TileDivision):
            print(Tilecounter)
            Tilecounter = Tilecounter + 1

            tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()

            subTileTree_Points,  _ = TileObj_MR.Classify_MultipleReturns(tile_segment_points)

            for t in subTileTree_Points:
                Trees_Buffer.append(t)
    
    end = time.time()
    MRtime = end - start
    print("MR Point Classification Algorithm Time : ",MRtime)
    
    print("Displaying Tree points")
    View3Dpoints(Trees_Buffer)

    #Write to LAS FILE
    util.Write_lasFile(Trees_Buffer, "MR_Vegetation_from_"+str(las_fileID), [0]*len(Trees_Buffer))

    script_end = time.time()
    script_time = script_end - script_start
    print("Script Time Time : ",script_time)




