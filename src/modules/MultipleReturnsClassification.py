#import open3d as o3d
import numpy as np
from sklearn import decomposition
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import json
from pyproj import Transformer



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
    
    def isPlane(self, XYZ, threshold):
        ''' 
            XYZ is n x 3 metrix storing xyz coordinates of n points
            It uses PCA to check the dimensionality of the XYZ
            th is the threshold, the smaller, the more strict for being 
            planar/linearity

            return 0 ==> randomly distributed
            return 1 ==> plane
            return 2 ==> line

        '''
        th = threshold #modified from 2e-3

        pca = decomposition.PCA()
        pca.fit(XYZ)
        pca_r = pca.explained_variance_ratio_
        t = np.where(pca_r < th)

        return t[0].shape[0]
        

    def Classify_MultipleReturns(self, MR_rawPoints, hp_eps=1.5, hp_min_points=30, HPF_THRESHOLD=200, PCA_PlaneTh = 2e-3):
        
        #Store Classified Tree points
        Tree_points = []
        Not_Tree_points = []

        db = DBSCAN(eps=hp_eps, min_samples=hp_min_points).fit(MR_rawPoints)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels_dbscan = db.labels_

        # #Open3d point cloud object
        # pcd = o3d.geometry.PointCloud()
        # #convert to vector3d object
        # MR_rawpointsVectorObj = o3d.utility.Vector3dVector(MR_rawPoints)
        # #store in pcd object
        # pcd.points = MR_rawpointsVectorObj

        #perform dbscan
        #labels_dbscan = np.array(pcd.cluster_dbscan(eps=hp_eps, min_points=hp_min_points))

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
                clustered_points = np.asarray(MR_rawPoints)
                #get points of latest outlier object
                labels_PC_points_reduced = list(clustered_points[interested_label_indexes])
                
                #check if cluster is planar - last check using PCA to ensure no planar structure included
                if self.isPlane(labels_PC_points_reduced,threshold=PCA_PlaneTh) == 0:#cluster points do not form a plane
                    for k in labels_PC_points_reduced:
                        Tree_points.append(k)
                else:
                    for m in labels_PC_points_reduced:
                        Not_Tree_points.append(m)
        
        return Tree_points, Not_Tree_points
    
    #convert x,y to lat long values of Trees found
    def ConvertLatLong(self,TreeIdentifier): # converting to ft ( 2263 - 4326)
        # LatLongCoords = []
        # for TreeLocation in TreeIdentifier:
        #     x = TreeLocation[0]
        #     y = TreeLocation[1]
        #     transformer = Transformer.from_crs("epsg:2263", "epsg:4326")
        #     lat, lon = transformer.transform(x,y)

        x = TreeIdentifier[0]*3.28
        y = TreeIdentifier[1]*3.28
        transformer = Transformer.from_crs("epsg:2263", "epsg:4326")
        lat, lon = transformer.transform(x,y)
        return [lat,lon]

    # # Count Number of Trees
    # def TreeCount(self,points):
    #     """
    #     Added
    #     ###################################################
    #     Counting number of Trees in DBSCAN cluster
    #     ###################################################
    #     """
    #     #Count number of Trees in Cluster
    #     X = np.array(points)[:,0]
    #     Y = np.array(points)[:,1]
    #     Z = np.array(points)[:,2]

    #     peaks = find_peaks(Z, height = 1.2,width=12,distance=1)
    #     peak_heights = peaks[1]['peak_heights'] #list of heights of peaks
    #     peak_pos = peaks[0]

    #     #Raw point cloud data
    #     Tree_LM = np.array((
    #         X[peak_pos], # convert ft to m
    #         Y[peak_pos], #convert ft to m
    #         peak_heights
    #     )).transpose()

    #     #RUN DBSCAN on identified peaks to cluster them
    #     # IDEA - Clustered peaks belong to a single Tree

    #     # from sklearn.cluster import DBSCAN
    #     # labels_dbscan_LMpoints = DBSCAN(eps=1.4, min_samples=1).fit(Tree_LM)
    #     TreeIdentifier = [] # stores a list of shape Nx3 - each element represents a Tree location
    #     if Tree_LM.shape[0] != 0 :

    #         db = DBSCAN(eps=1.4, min_samples=5).fit(Tree_LM)
    #         core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #         core_samples_mask[db.core_sample_indices_] = True
    #         labels_dbscan_LMpoints = db.labels_

    #         # pcd_LMpoints = o3d.geometry.PointCloud()
    #         # pcd_LMpoints.points = o3d.utility.Vector3dVector(Tree_LM)
    #         # labels_dbscan_LMpoints = np.array(pcd_LMpoints.cluster_dbscan(eps=1.4, min_points=5))

    #         labels_unique_LMpoints , label_counts_LMpoints = np.unique(labels_dbscan_LMpoints,return_counts=True)
    #         label_count_arr_LMpoints = np.asarray([labels_unique_LMpoints , label_counts_LMpoints]).T
            
    #         for i in range(len(labels_unique_LMpoints)):
    #             interested_cluster_label_LMpoints = labels_unique_LMpoints[i]
                            
    #             interested_label_indexes_LMpoints = np.where(labels_dbscan_LMpoints == interested_cluster_label_LMpoints)
    #             # need to use asarray, to extract points based on indexes later
    #             #clustered_LMpoints = np.asarray(pcd_LMpoints.points)
    #             clustered_LMpoints = np.asarray(Tree_LM)
    #             #get points of latest outlier object
    #             LM_points_cluster = clustered_LMpoints[interested_label_indexes_LMpoints]

    #             centroid_LMpoints = np.mean(LM_points_cluster,axis=0)
                
    #             TreeIdentifier.append(centroid_LMpoints.reshape(-1))
            
    #         TreeIdentifier = np.array(TreeIdentifier) # Stores number of Trees found in cluster with their locations
            
    #     return TreeIdentifier
    
    

    # def Get_MultipleReturnTreeCLusters(self, 
    #         MR_rawPoints, 
    #         Tree_Census_KDTree,TreeMapping_tolerance_thresh,trees_reduced_df,
    #         Tilecounter,
    #         las_filename,
    #         JSON_data_buffer,
    #         TreeClusterID,
    #         Ground_Tile_Zvalue,
    #         hp_eps=1.5, hp_min_points=30, 
    #         HPF_THRESHOLD=200, 
    #         PCA_PlaneTh = 2e-3):
        
        
    #     #Store Classified Tree points
    #     Tree_points = []
    #     Not_Tree_points = []

    #     db = DBSCAN(eps=1.4, min_samples=5).fit(MR_rawPoints)
    #     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #     core_samples_mask[db.core_sample_indices_] = True
    #     labels_dbscan = db.labels_

    #     # #Open3d point cloud object
    #     # pcd = o3d.geometry.PointCloud()
    #     # #convert to vector3d object
    #     # MR_rawpointsVectorObj = o3d.utility.Vector3dVector(MR_rawPoints)
    #     # #store in pcd object
    #     # pcd.points = MR_rawpointsVectorObj

    #     #perform dbscan
    #     # labels_dbscan = np.array(pcd.cluster_dbscan(eps=hp_eps, min_points=hp_min_points))

    #     #Stored label ID and count of labels for this cluster in 2d array
    #     labels_unique , label_counts = np.unique(labels_dbscan,return_counts=True)
    #     label_count_arr = np.asarray([labels_unique , label_counts]).T

    #     #HPF
    #     #Filter Tree Clouds by brute force approach (minimum number of points to represent a Tree)
    #     minimum_points_Tree_Cloud = HPF_THRESHOLD
    #     Potential_TreeLabels = []
    #     for x in range(len(label_count_arr)):
    #         if label_count_arr[x][1] > minimum_points_Tree_Cloud:
    #             Potential_TreeLabels.append(label_count_arr[x][0])
        
    #     labels = labels_dbscan
    #     for i in range(len(labels)):
    #         if labels[i] not in Potential_TreeLabels:
    #             #set label of unwanted(less that HPF threshold) points to -1 
    #             labels[i] = -1
    
    #     for i in range(len(Potential_TreeLabels)):
    #         if Potential_TreeLabels[i] == -1 :
    #             continue #do nothing for now
    #         else:
    #             #Remove Errorneous Trees in MR points

    #             #get cluster
    #             interested_cluster_label = Potential_TreeLabels[i]
    #             interested_label_indexes = np.where(labels == interested_cluster_label)
    #             # need to use asarray, to extract points based on indexes later
    #             clustered_points = np.asarray(MR_rawPoints)
    #             #get points of latest outlier object
    #             labels_PC_points_reduced = list(clustered_points[interested_label_indexes])
                
    #             #check if cluster is planar - last check using PCA to ensure no planar structure included
    #             if self.isPlane(labels_PC_points_reduced,threshold=PCA_PlaneTh) == 0:#cluster points do not form a plane
    #                 for k in labels_PC_points_reduced:
    #                     Tree_points.append(k)

    #                 # Once cluster has been designated to be part of a Tree
    #                 # Count the number of Trees and create JSON file

    #                 #Store number of Trees
    #                 TreeIdentifier = self.TreeCount(labels_PC_points_reduced)
    #                 #Convert X,Y,Z locations to lat,long values
    #                 #TreeIdentifierlatlong.append(ConvertLatLong(TreeIdentifier))

    #                 #Seperate Trees if multiple Trees found in each cluster
    #                 #Seperation is performed using Gaussian Mixture Models
    #                 if len(TreeIdentifier) == 0:
    #                     gmm_subset_components = 1
    #                 else:
    #                     gmm_subset_components = len(TreeIdentifier)
                    
    #                 gm_temp = GaussianMixture(n_components=gmm_subset_components, random_state=0).fit_predict(labels_PC_points_reduced)

    #                 #pseudo indivdual tree labels
    #                 # for gm_label_index in range(len(gm_temp)):
    #                 #     TreeGMM_labelArr.append(gm_temp[gm_label_index])

    #                 #Seperating each GMM Cluster
    #                 gm_labels_unique , gm_label_counts = np.unique(gm_temp,return_counts=True)
    #                 gm_label_count_arr = np.asarray([gm_labels_unique , gm_label_counts]).T

    #                 #Check if above threshold
    #                 gm_keep_labels = []
    #                 for x in range(len(gm_label_count_arr)):
    #                     if gm_label_count_arr[x][1] > minimum_points_Tree_Cloud:
    #                         gm_keep_labels.append(gm_label_count_arr[x][0])

    #                 gm_labels = gm_temp
    #                 for gm_label_index in range(len(gm_keep_labels)):
    #                     gm_interested_cluster_label = gm_keep_labels[gm_label_index]
    #                     gm_interested_label_indexes = np.where(gm_labels == gm_interested_cluster_label)
    #                     # need to use asarray, to extract points based on indexes later
    #                     gm_clustered_points = np.asarray(labels_PC_points_reduced)
    #                     #get points of latest outlier object
    #                     Estimated_SingleTreePoints = gm_clustered_points[gm_interested_label_indexes]

    #                     ########################################
    #                     # CREATE DICT TO APPEND TO JSON BUFFER

    #                     TreeClusterID = TreeClusterID + 1
    #                     GroundZValue = Ground_Tile_Zvalue
    #                     TreeClusterCentroid = np.mean(Estimated_SingleTreePoints, axis=0)
    #                     latitude , longitude = self.ConvertLatLong(TreeClusterCentroid)

    #                     Curr_Tree_Predloc = [TreeClusterCentroid[0],TreeClusterCentroid[1]]

    #                     dist , index = Tree_Census_KDTree.query(Curr_Tree_Predloc)

    #                     if(dist < TreeMapping_tolerance_thresh): #closest neighbour
    #                         Closest_Tree_index = index
    #                         Closest_Tree_ID = trees_reduced_df.tree_id.to_numpy()[Closest_Tree_index]
    #                     else:
    #                         #print("No Tree Found")
    #                         Closest_Tree_index = -1 #not found
    #                         Closest_Tree_ID = -1
    

    #                     #Generate Convex hull for this cluster
    #                     pts = np.array(Estimated_SingleTreePoints)
    #                     hull = ConvexHull(pts)
                        
    #                     # Convex Hull Data
    #                     ConvexHullData = {
    #                         "vertices" : hull.vertices.tolist(),
    #                         "simplices" : hull.simplices.tolist(),
    #                         "ClusterPoints" : hull.points.tolist(),
    #                         "equations" : hull.equations.tolist(),
    #                         "volume" : hull.volume,
    #                         "area" : hull.area
    #                     }

    #                     JSON_SingleTreeCluster_buffer = {
    #                         "TILEID" : Tilecounter,
    #                         "ClusterID" : str(las_filename)+"_"+str(TreeClusterID), 
    #                         "TreeFoliageHeight" : hull.points[:,2].max() - hull.points[:,2].min(),
    #                         "GroundZValue" : GroundZValue,
    #                         "ClusterCentroid" : TreeClusterCentroid.tolist(),
    #                         "TreeCountPredict": len(TreeIdentifier), #If >1 This Tree is part of a larger classified cluster
    #                         "ConvexHullDict" : ConvexHullData,
    #                         "PredictedTreeLocation" : {
    #                             "Latitude" : latitude,
    #                             "Longitude" : longitude
    #                         },
    #                         "Tree_Census_mapped_data" : {
    #                             'ClosestTreeIndex' : int(Closest_Tree_index),
    #                             'TreeCensusID' : int(Closest_Tree_ID)
    #                         }
    #                     }

    #                     JSON_data_buffer["MR_TreeClusterDict"].append(JSON_SingleTreeCluster_buffer)                        

    #             else:
    #                 for m in labels_PC_points_reduced:
    #                     Not_Tree_points.append(m)
        
    #     return Tree_points, Not_Tree_points, TreeClusterID
    
    # def Get_SingleTreeStrucutres(self, 
    #         MR_rawPoints, 
    #         Tree_Census_KDTree,TreeMapping_tolerance_thresh,
    #         Tilecounter,
    #         las_filename,
    #         JSON_data_buffer,
    #         TreeClusterID,
    #         Ground_Tile_Zvalue,
    #         hp_eps=1.5, hp_min_points=30, 
    #         HPF_THRESHOLD=200, 
    #         PCA_PlaneTh = 2e-3):
        
        
    #     #Store Classified Tree points
    #     Tree_points = []
    #     Not_Tree_points = []

    #     db = DBSCAN(eps=1.4, min_samples=5).fit(MR_rawPoints)
    #     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #     core_samples_mask[db.core_sample_indices_] = True
    #     labels_dbscan = db.labels_

    #     # #Open3d point cloud object
    #     # pcd = o3d.geometry.PointCloud()
    #     # #convert to vector3d object
    #     # MR_rawpointsVectorObj = o3d.utility.Vector3dVector(MR_rawPoints)
    #     # #store in pcd object
    #     # pcd.points = MR_rawpointsVectorObj

    #     #perform dbscan
    #     # labels_dbscan = np.array(pcd.cluster_dbscan(eps=hp_eps, min_points=hp_min_points))

    #     #Stored label ID and count of labels for this cluster in 2d array
    #     labels_unique , label_counts = np.unique(labels_dbscan,return_counts=True)
    #     label_count_arr = np.asarray([labels_unique , label_counts]).T

    #     #HPF
    #     #Filter Tree Clouds by brute force approach (minimum number of points to represent a Tree)
    #     minimum_points_Tree_Cloud = HPF_THRESHOLD
    #     Potential_TreeLabels = []
    #     for x in range(len(label_count_arr)):
    #         if label_count_arr[x][1] > minimum_points_Tree_Cloud:
    #             Potential_TreeLabels.append(label_count_arr[x][0])
        
    #     labels = labels_dbscan
    #     for i in range(len(labels)):
    #         if labels[i] not in Potential_TreeLabels:
    #             #set label of unwanted(less that HPF threshold) points to -1 
    #             labels[i] = -1
    
    #     for i in range(len(Potential_TreeLabels)):
    #         if Potential_TreeLabels[i] == -1 :
    #             continue #do nothing for now
    #         else:
    #             #Remove Errorneous Trees in MR points

    #             #get cluster
    #             interested_cluster_label = Potential_TreeLabels[i]
    #             interested_label_indexes = np.where(labels == interested_cluster_label)
    #             # need to use asarray, to extract points based on indexes later
    #             clustered_points = np.asarray(MR_rawPoints)
    #             #get points of latest outlier object
    #             labels_PC_points_reduced = list(clustered_points[interested_label_indexes])
                
    #             #check if cluster is planar - last check using PCA to ensure no planar structure included
    #             if self.isPlane(labels_PC_points_reduced,threshold=PCA_PlaneTh) == 0:#cluster points do not form a plane
    #                 for k in labels_PC_points_reduced:
    #                     Tree_points.append(k)

    #                 # Once cluster has been designated to be part of a Tree
    #                 # Count the number of Trees and create JSON file

    #                 #Store number of Trees
    #                 TreeIdentifier = self.TreeCount(labels_PC_points_reduced)
    #                 #Convert X,Y,Z locations to lat,long values
    #                 #TreeIdentifierlatlong.append(ConvertLatLong(TreeIdentifier))

    #                 #Seperate Trees if multiple Trees found in each cluster
    #                 #Seperation is performed using Gaussian Mixture Models
    #                 if len(TreeIdentifier) == 0:
    #                     gmm_subset_components = 1
    #                 else:
    #                     gmm_subset_components = len(TreeIdentifier)
                    
    #                 gm_temp = GaussianMixture(n_components=gmm_subset_components, random_state=0).fit_predict(labels_PC_points_reduced)

    #                 #pseudo indivdual tree labels
    #                 # for gm_label_index in range(len(gm_temp)):
    #                 #     TreeGMM_labelArr.append(gm_temp[gm_label_index])

    #                 #Seperating each GMM Cluster
    #                 gm_labels_unique , gm_label_counts = np.unique(gm_temp,return_counts=True)
    #                 gm_label_count_arr = np.asarray([gm_labels_unique , gm_label_counts]).T

    #                 #Check if above threshold
    #                 gm_keep_labels = []
    #                 for x in range(len(gm_label_count_arr)):
    #                     if gm_label_count_arr[x][1] > minimum_points_Tree_Cloud:
    #                         gm_keep_labels.append(gm_label_count_arr[x][0])

    #                 gm_labels = gm_temp
    #                 for gm_label_index in range(len(gm_keep_labels)):
    #                     gm_interested_cluster_label = gm_keep_labels[gm_label_index]
    #                     gm_interested_label_indexes = np.where(gm_labels == gm_interested_cluster_label)
    #                     # need to use asarray, to extract points based on indexes later
    #                     gm_clustered_points = np.asarray(labels_PC_points_reduced)
    #                     #get points of latest outlier object
    #                     Estimated_SingleTreePoints = gm_clustered_points[gm_interested_label_indexes]

    #                     ########################################
    #                     # CREATE DICT TO APPEND TO JSON BUFFER

    #                     TreeClusterID = TreeClusterID + 1
    #                     GroundZValue = Ground_Tile_Zvalue
    #                     TreeClusterCentroid = np.mean(Estimated_SingleTreePoints, axis=0)
    #                     latitude , longitude = self.ConvertLatLong(TreeClusterCentroid)

    #                     Curr_Tree_Predloc = [TreeClusterCentroid[0],TreeClusterCentroid[1]]

    #                     dist , index = Tree_Census_KDTree.query(Curr_Tree_Predloc)

    #                     if(dist < TreeMapping_tolerance_thresh): #closest neighbour
    #                         Closest_Tree_index = index
    #                         Closest_Tree_ID = trees_reduced_df.tree_id.to_numpy()[Closest_Tree_index]
    #                     else:
    #                         #print("No Tree Found")
    #                         Closest_Tree_index = -1 #not found
    #                         Closest_Tree_ID = -1
    

    #                     #Generate Convex hull for this cluster
    #                     pts = np.array(Estimated_SingleTreePoints)
    #                     hull = ConvexHull(pts)
                        
    #                     # Convex Hull Data
    #                     ConvexHullData = {
    #                         "vertices" : hull.vertices.tolist(),
    #                         "simplices" : hull.simplices.tolist(),
    #                         "ClusterPoints" : hull.points.tolist(),
    #                         "equations" : hull.equations.tolist(),
    #                         "volume" : hull.volume,
    #                         "area" : hull.area
    #                     }

    #                     JSON_SingleTreeCluster_buffer = {
    #                         "TILEID" : Tilecounter,
    #                         "ClusterID" : str(las_filename)+"_"+str(TreeClusterID), 
    #                         "TreeFoliageHeight" : hull.points[:,2].max() - hull.points[:,2].min(),
    #                         "GroundZValue" : GroundZValue,
    #                         "ClusterCentroid" : TreeClusterCentroid.tolist(),
    #                         "TreeCountPredict": len(TreeIdentifier), #If >1 This Tree is part of a larger classified cluster
    #                         "ConvexHullDict" : ConvexHullData,
    #                         "PredictedTreeLocation" : {
    #                             "Latitude" : latitude,
    #                             "Longitude" : longitude
    #                         },
    #                         "Tree_Census_mapped_data" : {
    #                             'ClosestTreeIndex' : int(Closest_Tree_index),
    #                             'TreeCensusID' : int(Closest_Tree_ID)
    #                         }
    #                     }

    #                     JSON_data_buffer["MR_TreeClusterDict"].append(JSON_SingleTreeCluster_buffer)                        

    #             else:
    #                 for m in labels_PC_points_reduced:
    #                     Not_Tree_points.append(m)
        
    #     return Tree_points, Not_Tree_points, TreeClusterID
