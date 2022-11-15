import open3d as o3d

from . import LasFilePreprocessing as LFP

lasTileClass = LFP.lasTile

class MR_class(lasTileClass):

    def __init__(self,LiDAR_Dataframe, TileDivision) -> None:
        lasTileClass.__init__(self,LiDAR_Dataframe, TileDivision)

        # self.lidar_Dataframe = LiDAR_Dataframe
        # self.TileDivision = TileDivision

        # self.rows, self.cols = (self.TileDivision, self.TileDivision)
        # self.Matrix_Buffer =  [[0]*self.cols for _ in range(self.rows)]
        

    def Get_MultipleReturnsVegetation(self):
        print("Function not completed yet :)")
        
        pass

    def Get_MultipleReturnsNotVegetation(self):
        pass