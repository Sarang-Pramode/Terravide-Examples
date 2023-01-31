import pandas as pd
import numpy as np
import laspy
import json
import pptk
from math import *
from matplotlib import pyplot as plt
from matplotlib import path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import alphashape
from shapely.geometry import Point
from scipy.spatial import ConvexHull
from pysolar.solar import *
import datetime
import csv
import pytz
import time
import logging
import re
import os
from pyproj import Transformer


# from ipywidgets import IntProgress
# from IPython.display import display

from multiprocessing import Pool


#HARDCODED VALUES

COLOR_DICT = {
    "WHITE" : [255,255,255],
    "RED" : [1,0,0],
    "GREEN" : [0,1,0],
    "BLUE" : [0,0,1],
    "YELLOW" : [255,255,0],
    "LIGHT_BLUE" : [0,255,255],
    "PINK" : [255,0,255]
}

DATE_DICT = {
    "YEAR" : 2021,
    "MONTH" : 6,
    "DAY" : 21
}


def Create_Dataframe_fromLas(lasFilePath):
    las = laspy.read(lasFilePath)
    #point_format = las.point_format
    lidar_points = np.array((las.X,las.Y,las.Z,las.intensity,las.classification, las.return_number, las.number_of_returns)).transpose()
    lidar_df = pd.DataFrame(lidar_points)
    lidar_df[0] = lidar_df[0]/100
    lidar_df[1] = lidar_df[1]/100
    lidar_df[2] = lidar_df[2]/100
    lidar_df.columns = ['X', 'Y', 'Z', 'intens', 'class', 'return_number', 'number_of_returns']
    
    return lidar_df

def Extract_SRdata(SR_JSONarr):
    #Store all Raw Treepoints
    SR_RawTreePoints = []
    #Store all Tree points mapped to a Tree ID
    SR_TreeClusterDict = {}
    #Store Tree Heights
    TreeHeights_ClusterHashmap = {}
    #Store GroundZ Elevation
    GroundElevation_ClusterHashmap = {}
    #Store All Tree Ids
    JSONClusterID_SR = []

    for i in range(len(SR_JSONarr)):
        #get points and cluster ID
        points = SR_JSONarr[i]["SRpointsInfo"]["SpecificClusterSRpoints"]
        cluster_id = SR_JSONarr[i]['SRpointsInfo']['SRpointsTreeCluster']
        #Storing with Cluster ID hashmap prevents storing multiple points from the same tree, if Tree present
        if cluster_id not in SR_TreeClusterDict:
            SR_TreeClusterDict[cluster_id] = points
        else:
            for p in points: 
                SR_TreeClusterDict[cluster_id].append(p)
        #Tree Heights
        if cluster_id not in TreeHeights_ClusterHashmap:
            TreeHeights_ClusterHashmap[cluster_id] = SR_JSONarr[i]['TreeFoliageHeight']
        #GroundZvalue
        if cluster_id not in GroundElevation_ClusterHashmap:
            GroundElevation_ClusterHashmap[cluster_id] = SR_JSONarr[i]['GroundZValue']

        #Raw Points
        for j in points:
            SR_RawTreePoints.append(j)

    return SR_RawTreePoints, SR_TreeClusterDict, TreeHeights_ClusterHashmap, GroundElevation_ClusterHashmap

def Extract_MRdata(MR_JSONarr):
    ##Store all Raw Treepoints
    MR_RawTreePoints = []
    #Store all Tree points mapped to a Tree ID
    MR_TreeClusterDict = {}
    #Store Cluster Location
    TreeLocation_ClusterHashmap = {}
    
    for i in range(len(MR_JSONarr)):
        #get points and cluster ID
        points = MR_JSONarr[i]["ConvexHullDict"]["ClusterPoints"]
        cluster_id = MR_JSONarr[i]['ClusterID']
        #Storing with Cluster ID hashmap prevents storing multiple points from the same tree, if Tree present
        if cluster_id not in MR_TreeClusterDict:
            MR_TreeClusterDict[cluster_id] = points
        else:
            for p_mr in points: 
                MR_TreeClusterDict[cluster_id].append(p_mr)

        Tree_lat = MR_JSONarr[i]["PredictedTreeLocation"]["Latitude"]
        Tree_long = MR_JSONarr[i]["PredictedTreeLocation"]["Longitude"]

        if cluster_id not in TreeLocation_ClusterHashmap:
            TreeLocation_ClusterHashmap[cluster_id] = [Tree_lat,Tree_long]

        #Raw points
        for k in points:
            MR_RawTreePoints.append(k)
    
    return MR_RawTreePoints, MR_TreeClusterDict, TreeLocation_ClusterHashmap

def Get_SR_Treepoints_from_Hashmap(TreeClusterID, SR_TreeClusterDict, las_filename='25192'):
    return SR_TreeClusterDict[las_filename+"_"+str(TreeClusterID)]

def Get_MR_Treepoints_from_Hashmap(TreeClusterID, MR_TreeClusterDict, las_filename='25192'):
    return MR_TreeClusterDict[las_filename+"_"+str(TreeClusterID)]

def Get_FullTree_points(TreeClusterArr, SR_TreeClusterDict, MR_TreeClusterDict, Ground_Elevation = 0, Tree_Height = 0):
    Full_Tree = []
    for Tree_Id in TreeClusterArr:
        TreePoints_SR = Get_SR_Treepoints_from_Hashmap(Tree_Id,SR_TreeClusterDict)
        TreePoints_MR = Get_MR_Treepoints_from_Hashmap(Tree_Id,MR_TreeClusterDict)
        All_SingleTree_points = np.concatenate((TreePoints_MR,TreePoints_SR), axis=0)
        All_SingleTree_points = All_SingleTree_points*3.28 # Convert m to ft
        for p in All_SingleTree_points:
            Full_Tree.append(p - [0,0,Ground_Elevation+Tree_Height]) #Height Adjusted, default = 0
            
    return np.array(Full_Tree)

def Get_FullTreeCentroid(Full_Tree_points):
    centroid = np.mean(Full_Tree_points,axis=0)
    centroid[2] = 0

    return centroid

def Get_Shadow(points, az, amp):
    projected_points = []
    for point in points:
        sinAz = sin( radians( az + 180.0 ) )
        cosAz = cos( radians( az + 180.0 ) )
        tanAmp = tan( radians(amp) )
        pointGroundX = point[0] + ( ( point[2] / tanAmp ) *sinAz )
        pointGroundY = point[1] + ( ( point[2] / tanAmp ) *cosAz )
        pointGroundZ =  point[2] * 0
    
        projected_points.append([pointGroundX,pointGroundY,pointGroundZ])
    
    return np.array(projected_points)

def Get_ShadowCharacteristics(Full_Tree, TreeShadow):

    F_distance, F_point = Get_FurthestPointFromTreeCentroid(Full_Tree, TreeShadow)
    centroid = Get_FullTreeCentroid(Full_Tree)
    hull = ConvexHull(TreeShadow[:,:2])

    Shadow_Area = hull.area
    Shadow_length = np.linalg.norm(centroid - F_point) # Tree Centroid - Furthest point from Tree centroid on Tree shadow
    shadow_breadth = Shadow_Area/Shadow_length

    return Shadow_length, shadow_breadth, Shadow_Area

def Get_FurthestPointFromTreeCentroid(Full_Tree, TreeShadow):
    
    Shadow_KDTree = KDTree(TreeShadow)
    centroid = Get_FullTreeCentroid(Full_Tree)
    dist,ind = Shadow_KDTree.query(centroid.reshape(1,3),len(TreeShadow))
    
    Furtherst_Point_Distance = dist[0][-1]
    Furthest_Point_Vector = TreeShadow[ind[0][-1]]

    return Furtherst_Point_Distance, Furthest_Point_Vector

def Get_ShadowAlphaShape(ShadowPoints):
    return alphashape.alphashape(ShadowPoints, alpha=0)

def Get_TreeLocation(TreeID, TreeID_Location_dict, las_filename='25192'):
    return TreeID_Location_dict[las_filename+"_"+str(TreeID)]

def Get_TreeGroundElevation(TreeID, GroundElevation_ClusterDict, las_filename='25192'):
    return GroundElevation_ClusterDict[las_filename+"_"+str(TreeID)]
    
def Get_TreeHeight(TreeID, TreeHeights_ClusterDict, las_filename='25192'):
    return TreeHeights_ClusterDict[las_filename+"_"+str(TreeID)]

def Get_GroundElevation(BuildingLidarDict):
    arr = (np.concatenate(list(BuildingLidarDict.values())))
    #Get ground elevation from building points min(list(BuildingLidarDict.values())[0][:,2])
    return min(arr[:,2])

#Code to automate Obtaining Az and Amp
def Get_SunData(Tree_Latitude, Tree_Longitude, year, month, day, hour, minute):
    date = datetime.datetime(year, month, day, hour, minute, 0, 0, tzinfo=datetime.timezone.utc)
    date_est = date.astimezone(pytz.timezone('US/Eastern')).isoformat()
    az = get_azimuth(Tree_Latitude, Tree_Longitude, date)
    amp = get_altitude(Tree_Latitude, Tree_Longitude, date)

    return date_est, az, amp

def pptk_wrapperSimplePlot(PointsArr, Color):
    v = pptk.viewer(PointsArr, [Color]*len(PointsArr))
    v.set(show_grid=False)
    v.set(show_axis=False)
    v.set(bg_color = [0,0,0,0])
    v.set(point_size = 0.04)

def sample_polygon(V, eps=0.25):
    # samples polygon V s.t. consecutive samples are no greater than eps apart
    # assumes last vertex in V is a duplicate of the first
    M = np.ceil(np.sqrt(np.sum(np.diff(V, axis=0) ** 2, axis = 1)) / eps)
    Q = []
    for (m, v1, v2) in zip(M, V[: -1], V[1:]):
        Q.append(np.vstack([ \
            np.linspace(v1[0], v2[0], endpoint = False), \
            np.linspace(v1[1], v2[1], endpoint = False)]).T)
    Q = np.vstack(Q)

    return Q

def Read_GeoJSON(filepath):
    with open('buildingsTile25192.geojson', 'rb') as fd:
        data = json.load(fd)

    return data

def Get_SampledBuildingFootprints(Buildingdata):
    BuildingFeature_Coords = [np.array(F['geometry']['coordinates'][0][0]) for F in Buildingdata['features']]
    SampledFootprint = np.vstack([sample_polygon(W) for W in BuildingFeature_Coords])
    SampledFootprint = np.c_[SampledFootprint, np.zeros(len(SampledFootprint))]

    return SampledFootprint

def Get_BuildingFootprint(GeoJSON_Filepath):
    BuidlingFootprintData = Read_GeoJSON(GeoJSON_Filepath)

    return Get_SampledBuildingFootprints(BuidlingFootprintData)

def Get_BuildingDataDict(BuildingFilePath, las_filename='25192', las_file_year=2015):
    with open('buildingsTile25192.geojson', 'rb') as fd:
        NYC_building_Footprint = json.load(fd)
    
    Vs_25192 = [np.array(F['geometry']['coordinates'][0][0]) for F in NYC_building_Footprint['features']]
    ### **Note** : NYC points already projected to state plane, No need for pyproj
    Ws_25192 = Vs_25192
    Total_Building_Count = len(NYC_building_Footprint['features'])
    #Building_Data_dict
    Building_coords_dict = {}
    B_ID = 0
    Building_Key = las_filename + "_" + str(las_file_year) + "_Building_" + str(B_ID) # initialized to 25192_2015_Building_0

    for building in range(Total_Building_Count):

        B_ID += 1
        Building_Key = las_filename + "_" + str(las_file_year) + "_Building_" + str(B_ID) # initialized to 25192_2015_Building_0
        building_coords = sample_polygon(Ws_25192[building]) #NYC_building_Footprint['features'][building]['geometry']['coordinates'][0][0]
        Building_coords_dict[Building_Key] = building_coords
    
    return Building_coords_dict

#NOTE: Function takes most time
def Get_BuildingsUnderEffectOfShadow(Building_coords_dict, TreeShadowPoints):
    ShadowAlphaShape = Get_ShadowAlphaShape(TreeShadowPoints)
    Affected_Building_IDarr = [] #Buildings Under the Direct Affect of the Shadow
    for B_ID, Sampled_B_coords in Building_coords_dict.items():
        bool_temp_list = []
        for point in Sampled_B_coords:
            tp = Point(point)
            bool_temp_list.append(tp.within(ShadowAlphaShape))
        #check if any points in the sampled footprint fall within the area of shadow
        if len(Sampled_B_coords[bool_temp_list]) > 0:
            Affected_Building_IDarr.append(B_ID)
    
    return Affected_Building_IDarr

def Get_BuildingsUnderEffectOfShadow_Rapid(Building_coords_dict, TreeShadowPoints):

    Affected_Building_IDarr = [] #Buildings Under the Direct Affect of the Shadow
    for B_ID, Sampled_B_coords in Building_coords_dict.items():
        bool_temp_list = []

        p = path.Path(TreeShadowPoints[:,:2])
        bool_temp_list = p.contains_points(Sampled_B_coords)
        # #check if any points in the sampled footprint fall within the area of shadow
        if len(Sampled_B_coords[bool_temp_list]) > 0:
            Affected_Building_IDarr.append(B_ID)
    
    return Affected_Building_IDarr

def Get_BuildingFootprintSampledCoords(BIDs_Arr, Building_coords_dict):

    labelled_BF_points = []
    for B_ID in BIDs_Arr:

        b_points = Building_coords_dict[B_ID]

        for p in b_points:
            labelled_BF_points.append(p)

    labelled_BF_points = np.array(labelled_BF_points) #shape : (Nx2)

    #reformat for plotting purposes
    LBF_arr = []
    for cord in range(len(labelled_BF_points)):
        x = labelled_BF_points[cord][0]
        y = labelled_BF_points[cord][1]
        z = 0
        LBF_arr.append([x,y,z])

    LBF_arr = np.array(LBF_arr)

    return LBF_arr

def Get_BuildingLidarPoints(BIDs_Arr, Building_coords_dict, lasdf):

    lidarPointsRaw = lasdf.iloc[:,:3].to_numpy()
    # Selecting Lidar Points within Building footprint

    X_max , X_min = lasdf.X.max(), lasdf.X.min()
    Y_max , Y_min = lasdf.Y.max(), lasdf.Y.min()

    X_plane_tile_divisor = 50 #in m - indicates the number of tiles you want to divide the tiles into
    Y_plane_tile_divisor = 50 #in m

    X_diff = X_max - X_min
    Y_diff = Y_max - Y_min

    X_div_len = X_diff/X_plane_tile_divisor
    Y_div_len = Y_diff/Y_plane_tile_divisor

    #NOTE : Iterating over all the points takes too long, need a more optimized way 

    lidar_BpointsDict = {}

    #For every building under the effect of the shadow
    for B_ID in BIDs_Arr:

        #List to append lidar points into for a specifc building
        B_ID_LidarPointsTempList = []

        #get the sampled building footprint points
        bf_points = Building_coords_dict[B_ID]

        #create a 2D convex Hull
        bf_shape = ConvexHull(bf_points[:,:2])

        #Get an area of points to look at (No need to iterate through all points in the tile)
        #get bounding subset to look at
        bf_shape_X_max = bf_shape.max_bound[0] + X_div_len
        bf_shape_Y_max = bf_shape.max_bound[1] + Y_div_len

        bf_shape_X_min = bf_shape.min_bound[0] - X_div_len
        bf_shape_Y_min = bf_shape.min_bound[1] - Y_div_len

        #Bounded_Vertices_BF = bf_points[bf_shape.vertices]

        lidar_subset_df = lasdf[
            (lasdf['X'].between(bf_shape_X_min, bf_shape_X_max, inclusive=False) &
        lasdf['Y'].between(bf_shape_Y_min, bf_shape_Y_max, inclusive=False))
        ]

        #Get only points from BF
        lidar_BFMasked_points = lidar_subset_df.iloc[:,:3].to_numpy()

        #Check if a lidar point is within a building Footprint 

        #Add a z-coordinate to BF ,  needed for alphashape
        bf_points_with_Z = np.c_[bf_points, np.zeros(len(bf_points))]
        #Create a shape of the building footprint
        bf_shape_alpha = alphashape.alphashape(bf_points_with_Z, alpha=0)

        for p in lidar_BFMasked_points:
            tp = Point(p)
            if(tp.within(bf_shape_alpha)):
                B_ID_LidarPointsTempList.append(p)

        if B_ID not in lidar_BpointsDict:
            lidar_BpointsDict[B_ID] = np.array(B_ID_LidarPointsTempList)

    return lidar_BpointsDict

def Set_BuildingLidarHeightAdjustment(BuidlingDataDict, height_Adjustment):
    
    for B_ID, B_Lidarpoints in BuidlingDataDict.items():

        bpArr = B_Lidarpoints
        #adjust height of buildings
        lidar_Bpoints_Hadj = []
        for bp in bpArr:
            lidar_Bpoints_Hadj.append(bp - [0,0,height_Adjustment])
        
        BuidlingDataDict[B_ID] = np.array(lidar_Bpoints_Hadj)
    
    return BuidlingDataDict


#https://github.com/ulikoehler/UliEngineering/blob/master/UliEngineering/Math/Coordinates.py
# -*- coding: utf-8 -*-
"""
Estimating BOunding Box
"""
__all__ = ["BoundingBox"]

class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, points):
        """
        Compute the upright 2D bounding box for a set of
        2D coordinates in a (n,2) numpy array.

        You can access the bbox using the
        (minx, maxx, miny, maxy) members.
        """
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a (n,2), array but it has shape {}".format(
                points.shape))
        if points.shape[0] < 1:
            raise ValueError("Can't compute bounding box for empty coordinates")
        self.minx, self.miny = np.min(points, axis=0)
        self.maxx, self.maxy = np.max(points, axis=0)

    @property
    def width(self):
        """X-axis extent of the bounding box"""
        return self.maxx - self.minx

    @property
    def height(self):
        """Y-axis extent of the bounding box"""
        return self.maxy - self.miny

    @property
    def area(self):
        """width * height"""
        return self.width * self.height

    @property
    def aspect_ratio(self):
        """width / height"""
        return self.width / self.height

    @property
    def center(self):
        """(x,y) center point of the bounding box"""
        return (self.minx + self.width / 2, self.miny + self.height / 2)

    @property
    def max_dim(self):
        """The larger dimension: max(width, height)"""
        return max(self.width, self.height)

    @property
    def min_dim(self):
        """The larger dimension: max(width, height)"""
        return min(self.width, self.height)

    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)
    #NOTE: Added by me
    def Get_Coords(self):
        return [self.minx, self.maxx, self.miny, self.maxy]
    

def Get_TreeShadowBoundingBoxLimits(TreeShadow):
    Box_obj = BoundingBox(TreeShadow[:,:2])
    Box_Bounds = np.array(Box_obj.Get_Coords())

    B_minx = Box_Bounds[0]
    B_maxx = Box_Bounds[1]
    B_miny = Box_Bounds[2]
    B_maxy = Box_Bounds[3]

    return B_minx, B_maxx, B_miny, B_maxy

def Get_TreeShadowBoundingBoxesCoords(TreeShadow, Distance):

    B_minx, B_maxx, B_miny, B_maxy = Get_TreeShadowBoundingBoxLimits(TreeShadow)

    Tree_ShadowBox_Coords = [
        [B_minx, B_miny, 0],
        [B_minx, B_maxy, 0],
        [B_maxx, B_maxy, 0],
        [B_maxx, B_miny, 0],
        [B_minx, B_miny, 0] # final one is to complete the box used for plotting purposes only
    ]

    Extend_dist = Distance #in m
    Tree_ShadowBoxExtend_Coords = [
        [B_minx, B_miny - Extend_dist, 0],
        [B_minx, B_maxy, 0],
        [B_maxx, B_maxy, 0],
        [B_maxx, B_miny - Extend_dist, 0],
        [B_minx, B_miny - Extend_dist, 0] # final one is to complete the box used for plotting purposes only
    ]

    return Tree_ShadowBox_Coords, Tree_ShadowBoxExtend_Coords

#Facade shade
def Get_FacadeShadePoints(TreeShadow, PrimaryBuilding_IdArr, Building_coords_dict):

    FacadeShade_Points = []

    RemainingShadow_Points = TreeShadow

    for B_ID in PrimaryBuilding_IdArr:

        #get the sampled building footprint points
        bf_points = Building_coords_dict[B_ID]

        #Get only points from BF
        BuildingFootprint_shape = alphashape.alphashape(bf_points, alpha=0)

        curr_BF_bool_list = []
        #find Tree Shadow Points in StreetShade
        for tsp in RemainingShadow_Points:
            tp = Point(tsp)
            curr_BF_bool_list.append(tp.within(BuildingFootprint_shape))
        
        for fp in RemainingShadow_Points[curr_BF_bool_list]: #only inside the BF shape
            FacadeShade_Points.append(fp)
        
        RemainingShadow_Points = RemainingShadow_Points[~np.array(curr_BF_bool_list)]
    
    return RemainingShadow_Points, FacadeShade_Points

def Get_InShadePoints(TreeShadow, All_Building_LidarPointsDict_HAdj, Az, Amp):

    InShade_Points = []
    RemainingShadow_Points = TreeShadow

    for B_ID, B_lidarpoints in All_Building_LidarPointsDict_HAdj.items():

        B_Shadow = Get_Shadow(B_lidarpoints, Az, Amp)

        #Get only points from BS
        BuildingShadow_shape = alphashape.alphashape(B_Shadow, alpha=0)

        curr_BS_bool_list = []
        #find Tree Shadow Points in InShade
        for tsp in RemainingShadow_Points:
            tp = Point(tsp)
            curr_BS_bool_list.append(tp.within(BuildingShadow_shape))
        
        for Ip in RemainingShadow_Points[curr_BS_bool_list]: #only inside the BS shape
            InShade_Points.append(Ip)
            
        if(len(curr_BS_bool_list) > 0):
            RemainingShadow_Points = RemainingShadow_Points[~np.array(curr_BS_bool_list)]
    
    return RemainingShadow_Points, InShade_Points

def InitiateShadingLogger(filename:str,year:int)-> None:

    LoggerPath = "Datasets/"+"Package_Generated/TEST_DIR/"+filename+"/"+str(year)+"/Shading_Logs_"+filename+"/"

    print("Logger Folder Path : ",LoggerPath)
    # Check whether the specified pptk_capture_path exists or not
    isExist = os.path.exists(LoggerPath)

    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(LoggerPath)

    logfilename = LoggerPath + filename+'.log' 
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=logfilename, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

def parse_log_Completedfiles(log_file_path):
    """_summary_

    Args:
        log_file_path (_type_): _description_

    Returns:
        _type_: _description_
    
    Usage:
    log_file_path = 'TEST_log.log'
    filenames = parse_log_Completedfiles(log_file_path)
    print(filenames)
    """
    filenames = []
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = re.search(r'(?<=Completed file: )\S+', line)
            if match:
                filenames.append(match.group())
    return filenames

def Get_dirnames(parent_path:str)->list:
    dirnames = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
    return dirnames


def ProcessShading(f,year,fpath):

    try: 
    
        stime = time.time()

        InitiateShadingLogger(f,year)

        logging.info("TerraVide lidar Shading Metrics Estimation Initated")

        las_file_path = 'Datasets/Package_Generated/'+f[:-4]+'/'+str(year)+'/LasClassified_'+f[:-4]+'/lasFile_Reconstructed_'+f[:-4]+'.las'

        logging.info("Reading Reconstructed lasfile from : %s",las_file_path)

        #Create a dataframe from las file
        lasdf = Create_Dataframe_fromLas(las_file_path)

        

        etime = time.time()



    except Exception as e:
        logging.error("ERROR : %s",str(e))






if __name__ == "__main__":
    #TEST SCRIPT - LOGS in TEST Directory - modify to Dataset paths with original files for deployment

    script_start_time = time.time()

    #Get las directory names from parent root folder
    p_rootpath = "Datasets/Package_Generated/" #'/Volumes/Elements/TerraVide/Datasets/FTP_files/LiDAR/'
    year = 2017

    LAS_dirnames = Get_dirnames(p_rootpath)

    print("FileCount : "+str(len(LAS_dirnames))+" .las files found in path = "+p_rootpath )

    args = [(i, year,p_rootpath) for i in LAS_dirnames[:2]]#testing for first 2 files

    for arg_item in args:
        print(arg_item)

    with Pool(2) as p:

         p.starmap(ProcessShading,args)
        
    script_end_time = time.time()

    print("SCRIPT TOTAL TIME (in min): ",(script_end_time - script_start_time)/60)



