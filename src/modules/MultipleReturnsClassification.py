import open3d as o3d
import numpy as np
from sklearn import decomposition


#custom modules
from . import LasFilePreprocessing as LFP

lasTileClass = LFP.lasTile

class MR_class(lasTileClass):

    def __init__(self,LiDAR_Dataframe, TileDivision) -> None:
        lasTileClass.__init__(self,LiDAR_Dataframe, TileDivision)

        #Populate Subtile Array Buffer if not called
        if not self.Matrix_BufferFilled :
            #print("WARN : Filling Matrix_Buffer as user did not call")
            self.localMatrixBuffer = self.Get_subtileArray()
        
        self.localMatrixBuffer = self.Matrix_Buffer

        # self.lidar_Dataframe = LiDAR_Dataframe
        # self.TileDivision = TileDivision

        # self.rows, self.cols = (self.TileDivision, self.TileDivision)
        # self.Matrix_Buffer =  [[0]*self.cols for _ in range(self.rows)]
    
    def isPlane(self, XYZ):
        ''' 
            XYZ is n x 3 metrix storing xyz coordinates of n points
            It uses PCA to check the dimensionality of the XYZ
            th is the threshold, the smaller, the more strict for being 
            planar/linearity

            return 0 ==> randomly distributed
            return 1 ==> plane
            return 2 ==> line

        '''
        th = 2e-3 #modified from 2e-3

        pca = decomposition.PCA()
        pca.fit(XYZ)
        pca_r = pca.explained_variance_ratio_
        t = np.where(pca_r < th)

        return t[0].shape[0]
        

    def Classify_MultipleReturns(self, MR_rawPoints, hp_eps=1.5, hp_min_points=30, HPF_THRESHOLD=200):
        
        #Store Classified Tree points
        Tree_points = []
        Not_Tree_points = []

        #Open3d point cloud object
        pcd = o3d.geometry.PointCloud()
        #convert to vector3d object
        MR_rawpointsVectorObj = o3d.utility.Vector3dVector(MR_rawPoints)
        #store in pcd object
        pcd.points = MR_rawpointsVectorObj

        #perform dbscan
        labels_dbscan = np.array(pcd.cluster_dbscan(eps=hp_eps, min_points=hp_min_points))

        #Stored label ID and count of labels for this cluster in 2d array
        labels_unique , label_counts = np.unique(labels_dbscan,return_counts=True)
        label_count_arr = np.asarray([labels_unique , label_counts]).T

        #HPF
        #Filter Tree Clouds by brute force approach (minimum number of points to represent a Tree)
        minimum_points_Tree_Cloud = HPF_THRESHOLD
        Potential_TreeLabels = []
        for x in range(len(label_count_arr)):
            if label_count_arr[x][1] > minimum_points_Tree_Cloud:
                Potential_TreeLabels.append(label_count_arr[x][0])
        
        labels = labels_dbscan
        for i in range(len(labels)):
            if labels[i] not in Potential_TreeLabels:
                #set label of unwanted(less that HPF threshold) points to -1 
                labels[i] = -1
    
        for i in range(len(Potential_TreeLabels)):
            if Potential_TreeLabels[i] == -1 :
                continue #do nothing for now
            else:
                #Remove Errorneous Trees in MR points

                #get cluster
                interested_cluster_label = Potential_TreeLabels[i]
                interested_label_indexes = np.where(labels == interested_cluster_label)
                # need to use asarray, to extract points based on indexes later
                clustered_points = np.asarray(pcd.points)
                #get points of latest outlier object
                labels_PC_points_reduced = list(clustered_points[interested_label_indexes])
                
                #check if cluster is planar - last check using PCA to ensure no planar structure included
                if self.isPlane(labels_PC_points_reduced) == 0:#cluster points do not form a plane
                    for k in labels_PC_points_reduced:
                        Tree_points.append(k)
                else:
                    for m in labels_PC_points_reduced:
                        Not_Tree_points.append(m)
        
        return Tree_points, Not_Tree_points

    def Get_MultipleReturnsNotVegetation(self):
        pass