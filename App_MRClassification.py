import src.modules.MultipleReturnsClassification as MRC

import pptk

# Aliases for simplicity
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

    print("Multiple Returns Classifcation Algorithm")

    TileDivision = 10 #test purposes
    rows, cols = (TileDivision, TileDivision)

    lasfilepath = 'Datasets/FTP_files/LiDAR/NYC_2017/25192.las'
    
    #Read las file
    lasfile_object = LasHandling.Read_lasFile(lasfilepath)
    #Create Dataframe from lasfile
    lidar_df, rawpoints = LasHandling.Create_lasFileDataframe(lasfileObject=lasfile_object)

    #Extract MR and SR points from Dataframe
    MR_df = LasHandling.Get_MRpoints(lidar_df)
    SR_df = LasHandling.Get_SRpoints(lidar_df)

    #lasTile class
    TileObj = MRC.MR_class(SR_df,TileDivision)

    #Serialized Creation of Lidar Subtiles
    lidar_TilesubsetArr = TileObj.Get_subtileArray()

    #Get TreePoints
    MR_TreePoints = TileObj.Get_MultipleReturnsVegetation()



