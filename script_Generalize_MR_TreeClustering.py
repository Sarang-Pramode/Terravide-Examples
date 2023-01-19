import src.modules.utils as util
import random
import src.modules.MultipleReturnsClassification as MRC
import time
import pptk
import numpy as np
import seaborn as sns 
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from pyproj import Transformer
import laspy
import os
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score # How best can we seperate clusters

#record start time of script
script_start = time.time()

#Get list of files from ftp server
filenames = util.FTP_GetFileList(2017)

for f in filenames:

    print("Filename : ", f)

    BLACKLIST_FILEIDs = [10170,2260, 255, 30265]

    if int(f[:-4]) in BLACKLIST_FILEIDs:
        print("#### FILE INGORED - BLACKLISTED ####")
        continue

    # Download Las File to hard disk
    las_fileID = int(''.join(c for c in f if c.isdigit())) 
    #Testing : List in Hardisk - 15212, 25192, 47217, 47219, 987210 - Empire State, 45172 - Laguardia
    year = 2017

    Filename = f #str(las_fileID)+'.las'
    External_Disk_Path = '/Volumes/Elements/Terravide/Datasets/FTP_files/LiDAR/'
    util.FTP_download_lasfile(Filename,year,folderpath=External_Disk_Path)

    #Check if output exist

    CSV_loc = []
    CSV_SUBTILE_EPS_Arr = []

    FilesCompletedPath = '/Volumes/Elements/Terravide/Datasets/Package_Generated/MR_JSON_Files/2017/'
    Screenshots_Path = '/Volumes/Elements/TerraVide/Datasets/Package_Generated/PPTK_screenshots/'

    MR_JSONdir_list = os.listdir(FilesCompletedPath)

    ScreenCatpuredir_list = os.listdir(Screenshots_Path)
    
    #print("Files and directories in '", FilesCompletedPath, "' :")
    
    # prints all files
    #print(MR_JSONdir_list)

    las_fileID_Completed_MRjson = []
    las_fileID_Completed_ScreenCapture = []

    for completed_filename in MR_JSONdir_list:
        las_fileID_in_MR_JSONdir_list = ''.join(c for c in completed_filename if c.isdigit())[:-4]
        las_fileID_Completed_MRjson.append(las_fileID_in_MR_JSONdir_list)
    
    for completed_filename in ScreenCatpuredir_list:
        las_fileID_in_ScreenCapturedir_list = ''.join(c for c in completed_filename if c.isdigit())[:-4]
        las_fileID_Completed_ScreenCapture.append(las_fileID_in_ScreenCapturedir_list)
    


    if (f[:-4]) in las_fileID_Completed_ScreenCapture:
        print("Screenshot Completed for :",f)
        continue
    else:
        print("Screenshot not found in completed dir")
    

    #Object to handle las preprocessing
    LasHandling = MRC.LFP

    lasfilepath = External_Disk_Path+'NYC_'+str(year)+'/'+str(las_fileID)+'.las'

    #Read las file
    lasfile_object = LasHandling.Read_lasFile(lasfilepath)

    TileDivision = 12
    rows, cols = (TileDivision, TileDivision)

    #Create Dataframe from lasfile
    lidar_df, rawpoints = LasHandling.Create_lasFileDataframe(lasfileObject=lasfile_object)

    #Divide lidar_df into smaller portion for development
    portion_size = 100 # 1-100 %
    lidar_df = lidar_df.head(int(len(lidar_df)*(portion_size/100)))

    #sanity check
    #pptk.viewer(lidar_df.iloc[:,:3].to_numpy())

    #Taking a smaller portion of the lidar tile

    lidar_df, rawpoints = LasHandling.Create_lasFileDataframe(lasfileObject=lasfile_object)


    X_max_G , X_min_G = lidar_df.X.max(), lidar_df.X.min()
    Y_max_G , Y_min_G = lidar_df.Y.max(), lidar_df.Y.min()

    X_plane_tile_divisor_G = 60 #in m - indicates the number of tiles you want to divide the tiles into
    Y_plane_tile_divisor_G = 60 #in m

    X_diff_G = X_max_G - X_min_G
    Y_diff_G = Y_max_G - Y_min_G

    #X_div_len = math.ceil(X_diff_G/X_plane_tile_divisor_G) #introduced a margin of error at boundaries
    #Y_div_len = math.ceil(Y_diff_G/Y_plane_tile_divisor_G)
    X_div_len_G = X_diff_G/X_plane_tile_divisor_G
    Y_div_len_G = Y_diff_G/Y_plane_tile_divisor_G

    # Create 2D
    rows, cols = (X_plane_tile_divisor_G, Y_plane_tile_divisor_G)
    lidar_subset_df = [[0]*cols]*rows

    ###

    Potential_Ground_Points = []
    Other_points = []
    max_count = X_plane_tile_divisor_G*Y_plane_tile_divisor_G


    #Iterating through each tile
    Tilecounter = 0

    for row in range(X_plane_tile_divisor_G):
        for col in range(Y_plane_tile_divisor_G):
            Tilecounter = Tilecounter + 1
            print("Tile Number : ",Tilecounter," FILENAME : ", las_fileID)

            x_cloud_subset_min = X_min_G + row*X_div_len_G
            x_cloud_subset_max = X_min_G + row*X_div_len_G + X_div_len_G
            y_cloud_subset_min = Y_min_G + col*Y_div_len_G
            y_cloud_subset_max = Y_min_G + col*Y_div_len_G + Y_div_len_G
            lidar_subset_df[row][col] = lidar_df[ #Store each subset of the tile
                (lidar_df['X'].between(x_cloud_subset_min, x_cloud_subset_max, inclusive=False) &
            lidar_df['Y'].between(y_cloud_subset_min, y_cloud_subset_max, inclusive=False))
            ]

            #select rows from lidar subset df
            tile_segment_points = lidar_subset_df[row][col].iloc[:,:3].to_numpy()

            #Ground Seperation

            #Finding lowest points in the segemnt
            ground_H_thresh = 2 #height threshold in meters
            z_values = tile_segment_points[:,2]
            
            if (len(z_values) != 0):
                z_min = np.min(z_values)
                # #get K smallest values
                # smallest_K = np.sort(z_values)
                # #take avg
                # z_avg = np.mean(smallest_K)
                ground_H_thresh_perc = 0.1
                Z_Height_thresh = z_min*ground_H_thresh_perc

                lowest_points_idx = [idx for idx,record in enumerate(z_values) if record > (z_min) and record < (z_min+Z_Height_thresh) ]
                lowest_points = tile_segment_points[lowest_points_idx]

                Other_points_idx =  [idx for idx,record in enumerate(z_values) if record > (z_min+Z_Height_thresh) ]
                Not_ground_points = tile_segment_points[Other_points_idx]
                

                for k in lowest_points:
                    Potential_Ground_Points.append(k) #append points which may be potentially ground points
                for l in Not_ground_points:
                    Other_points.append(l)

    Potential_Ground_Points_array = np.array(Potential_Ground_Points)
    Potential_Other_Points_array = np.array(Other_points)

    if len(Potential_Ground_Points_array) == 0:
        Potential_Ground_Points_array = [[0,0,0]] #default

    GPoints_df = pd.DataFrame(Potential_Ground_Points_array , columns=['X','Y','Z'])
    NGPoints_df = pd.DataFrame(Potential_Other_Points_array , columns=['X','Y','Z'])

    Tree_Height_Limit = 80 # in m
    lidar_df_ZLimited  = lidar_df[lidar_df.Z < Tree_Height_Limit]

    lidar_df = lidar_df_ZLimited
    #pptk.viewer(lidar_df.iloc[:,:3].to_numpy())

    #Extract MR and SR points from Dataframe
    MR_df = LasHandling.Get_MRpoints(lidar_df)
    SR_df = LasHandling.Get_SRpoints(lidar_df)

    #lasTile class
    TileObj_SR = MRC.MR_class(SR_df,TileDivision) #Single Return Points
    TileObj_MR = MRC.MR_class(MR_df,TileDivision) #Multiple Return Points

    #Serialized Creation of Lidar Subtiles
    lidar_TilesubsetArr = TileObj_MR.Get_subtileArray()

    #Print Lat , Long
    ix, iy = np.mean(MR_df.X.to_numpy()), np.mean(MR_df.Y.to_numpy()) 

    transformer = Transformer.from_crs("epsg:2263", "epsg:4326")
    lat, lon = transformer.transform(ix*3.28, iy*3.28)
    print(str(lat)+","+str(lon))



    def Get_eps_NN_KneeMethod(cluster_df, N_neighbors = 12, display_plot=False):

        nearest_neighbors = NearestNeighbors(n_neighbors=N_neighbors)
        neighbors = nearest_neighbors.fit(cluster_df)
        distances, indices = neighbors.kneighbors(cluster_df)
        distances = np.sort(distances[:,N_neighbors-1], axis=0)

        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        if (display_plot):
            fig = plt.figure(figsize=(5, 5))
            knee.plot_knee()
            plt.xlabel("Points")
            plt.ylabel("Distance")
            print(distances[knee.knee])
        
        return distances[knee.knee]

    def Normalize_points(points):
        return points / np.linalg.norm(points)

    #Note : Not Used in Current workflow
    def Get_Optimal_MinSamples(points,ep,start=20,end=70,step=10):

        range_min_samples = np.arange(start, end, step)

        S_score_min_samples_hashmap = {} # S_score -> min_samples

        for m in range_min_samples :
                    
            db = DBSCAN(eps=ep, min_samples=m).fit(points)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            #print(set(labels))
            if(len(set(labels))) < 2 :
                return (1.5,0)
            silhouette_avg = silhouette_score(points, labels)
            print("For eps value = ",ep,"; Min Samples = ",m, "The average silhouette_score is :", silhouette_avg)

            if silhouette_avg not in S_score_min_samples_hashmap:
                S_score_min_samples_hashmap[silhouette_avg] = ep
            
            Best_min_samples = S_score_min_samples_hashmap[max(S_score_min_samples_hashmap)]
        
        return (Best_min_samples, max(S_score_min_samples_hashmap))
    
    All_eps = [] #Stores all eps values by tile id
    N_Neighbours = 12
    for row in range(TileDivision):
        for col in range(TileDivision):

            if(len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

                cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]

                tile_eps = Get_eps_NN_KneeMethod(cluster_df)

                All_eps.append(tile_eps)

    # fig = plt.figure(figsize=(10, 10))
    # plt.plot(All_eps)
    # plt.xlabel("Tile ID")
    # plt.ylabel("Eps value")

    Optimal_EPS = np.mean(All_eps)
    print("EPS : ",Optimal_EPS)
    
    CSV_loc.append([lat, lon])
    CSV_SUBTILE_EPS_Arr.append(All_eps)

    Spatial_HP_df = pd.DataFrame({"las_fileID" : las_fileID, "Tile_Location" : CSV_loc,
                    "EPS_variations" : CSV_SUBTILE_EPS_Arr})

    Spatial_HP_df.to_csv('Spatial_HP_Distribution.csv')

    #DO MR Treepoint extraction

    # Classify tree points in MR without extracting tree clusters

    Tilecounter = 0
    Trees_Buffer = []

    N_Neighbours = 12

    for row in range(TileDivision):
        for col in range(TileDivision):

            print('-'*40)
            
            print("TILE ID : ",Tilecounter)
            Tilecounter = Tilecounter + 1

            if (len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

                cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]

                tile_eps = Get_eps_NN_KneeMethod(cluster_df) #round(Optimal_EPS,2)
                print(tile_eps)

                tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()

                subTileTree_Points,  _ = TileObj_MR.Classify_MultipleReturns(tile_segment_points,tile_eps)

                for t in subTileTree_Points:
                    Trees_Buffer.append(t)
                
                print("Trees added")
            
            else:
                print("empty tileset found")

    Trees_Buffer = np.array(Trees_Buffer)

    #pptk.viewer(Trees_Buffer)

    #plotting tree poitns found
    p1 = Trees_Buffer
    p2 = MR_df.iloc[:,:3].to_numpy()
    All_points_1 = np.concatenate((p1, p2), axis=0)
    rgb_p2 =  [[1,0,0]]*len(p2) #Set red colour
    rgb_p1 = [[0,1,0]]*len(p1) #set green colour - Classified tree points
    All_rgb = np.concatenate((rgb_p1, rgb_p2,), axis=0)

    v = pptk.viewer(All_points_1, All_rgb)
    v.set(show_grid=False)
    v.set(show_axis=False)
    v.set(bg_color = [0,0,0,1])
    v.set(point_size = 0.04)


    x_mean = np.mean(p2[:,0])
    y_mean = np.mean(p2[:,1])
    z_mean = np.mean(p2[:,2])

    v.set(phi=np.pi/4)
    v.set(theta=np.pi/6)
    v.set(r=900)
    v.set(lookat=[x_mean,y_mean,0])

    pptk_capture_path = "/Volumes/Elements/TerraVide/Datasets/" + "Package_Generated/PPTK_screenshots/"
    v.capture(pptk_capture_path+"Capture"+str(las_fileID)+"_"+str(year)+'.png')
    time.sleep(1) #screenshots were not captured after v.close was added
    v.close()

    if (f[:-4]) in las_fileID_Completed_MRjson:
        print("Proccessed Clusters Completed for :",f)
        continue
    else:
        print("Proccessed Clusters not found in MR JSON dir")

    #TODO : Make a function for this

    # Tree Census
    Trees_found_in_Tile = True

    path = "/Volumes/Elements/Terravide/Datasets/NYC_Tree_Dataset/2015StreetTreesCensus_TREES.csv"
    Tree_Dataset = pd.read_csv(path)
    las = laspy.read(lasfilepath) # .las file taken from NYC topbathymetric 2017 Lidar data

    point_format = las.point_format
    lidarPoints = np.array((las.X,las.Y,las.Z,las.intensity,las.classification, las.return_number, las.number_of_returns)).transpose()
    lidar_dfRead = pd.DataFrame(lidarPoints)
    # correct XYZ scale to state plane
    lidar_dfRead[0] = lidar_dfRead[0]/100
    lidar_dfRead[1] = lidar_dfRead[1]/100
    lidar_dfRead[2] = lidar_dfRead[2]/100
    # find bounds of las file
    x_min = lidar_dfRead[0].min()
    x_max = lidar_dfRead[0].max()
    y_min = lidar_dfRead[1].min()
    y_max = lidar_dfRead[1].max()
    # select trees in lidar footprint in a new dataframe
    trees_df2 = Tree_Dataset.copy()
    trees_df2 = trees_df2[trees_df2['x_sp']>x_min]
    trees_df2 = trees_df2[trees_df2['x_sp']<x_max]
    trees_df2 = trees_df2[trees_df2['y_sp']>y_min]
    trees_reduced_df = trees_df2[trees_df2['y_sp']<y_max]

    True_TreeLoc = {"Tree_X" : trees_reduced_df.x_sp.to_numpy(),
                    "Tree_Y" : trees_reduced_df.y_sp.to_numpy()}

    # Used to Map Tree ID to Predicted Tree Cluster
    True_TreeLoc_X = True_TreeLoc['Tree_X']/3.28
    True_TreeLoc_Y = True_TreeLoc['Tree_Y']/3.28
    Tree_Census_Loc_xy = np.stack((True_TreeLoc_X,True_TreeLoc_Y),axis=1)

    #Use KDTree
    if (len(Tree_Census_Loc_xy) == 0):
        print("No Trees in Tree census")
        Trees_found_in_Tile = False

    from scipy import spatial

    if(Trees_found_in_Tile):
        Tree_Census_KDTree = spatial.KDTree(Tree_Census_Loc_xy)
    else:
        #TODO : QUICK FIX - FIX Later
        simpleFixarr = [[313223.43600915,  59044.84813689],
        [313260.9461372 ,  58974.44035335],
        [313212.84671951,  59072.36252561]]
        Tree_Census_KDTree = spatial.KDTree(simpleFixarr)

    Tilecounter = 0
    Trees_Buffer = []

    #JSON Buffer vars
    TreeClusterID = 0
    JSON_data_buffer = {
        "lasFileID" : las_fileID,
        "RecordedYear" : year,
        "MR_TreeClusterDict" : []
    }

    #For Ground - TODO: write into library

    X_max , X_min = lidar_df.X.max(), lidar_df.X.min()
    Y_max , Y_min = lidar_df.Y.max(), lidar_df.Y.min()

    X_diff = X_max - X_min
    Y_diff = Y_max - Y_min

    X_div_len = X_diff/TileDivision
    Y_div_len = Y_diff/TileDivision

    rows, cols = (TileDivision, TileDivision)
    lidar_subsetGround_df = [[0]*cols]*rows

    for row in range(TileDivision):
        for col in range(TileDivision):

            print('-'*40)
            
            print("TILE ID : ",Tilecounter)
            print("TreeCLusterID : ",TreeClusterID)
            Tilecounter = Tilecounter + 1

            cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]

            if (len(cluster_df.to_numpy()) > N_Neighbours):

                tile_eps = Get_eps_NN_KneeMethod(cluster_df) #round(Optimal_EPS,2)
                print("EPS :",tile_eps)

                tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()

                #TODO : Clean code for Ground Z value

                        #Ground
                lidar_subsetGround_df[row][col] = GPoints_df[ #Store each subset of the tile
                    (GPoints_df['X'].between(x_cloud_subset_min, x_cloud_subset_max, inclusive=False) &
                GPoints_df['Y'].between(y_cloud_subset_min, y_cloud_subset_max, inclusive=False))
                ]

                #select ground tile segment points
                tile_segment_pointsGround = lidar_subsetGround_df[row][col].iloc[:,:3].to_numpy()
                Ground_Tile_Zvalue = np.mean(tile_segment_pointsGround[:,2])


                subTileTree_Points, _ , TreeCounterID = TileObj_MR.Get_MultipleReturnTreeCLusters(
                                            tile_segment_points,
                                            Tree_Census_KDTree,5,trees_reduced_df, #tree mapping tolerance thresh = 5 m
                                            Tilecounter,
                                            las_fileID,
                                            JSON_data_buffer,
                                            TreeClusterID,
                                            Ground_Tile_Zvalue,
                                            tile_eps)

                TreeClusterID = TreeCounterID #TODO : make global var 

                for t in subTileTree_Points:
                    Trees_Buffer.append(t)
            
            print("Trees Clusters added to JSON")

    Trees_Buffer = np.array(Trees_Buffer)

    #pptk.viewer(Trees_Buffer)

    JSONfoldernName = "MR_JSON_Files"

    jpath = "/Volumes/Elements/TerraVide/Datasets/" + "Package_Generated/" + JSONfoldernName + "/" + str(year)

    # Check whether the specified jpath exists or not
    isExist = os.path.exists(jpath)

    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(jpath)

    with open(jpath+"/"+str(las_fileID)+"_"+str(year)+"_TreeCluster.json", "w") as jsonFile:
        jsonFile.truncate(0)
        json.dump(JSON_data_buffer, jsonFile)

    CSVfoldernName = "MR_CSV_Files"

    csvpath = "/Volumes/Elements/TerraVide/Datasets/" + "Package_Generated/" + CSVfoldernName + "/" + str(year)

    # Check whether the specified jpath exists or not
    isExist = os.path.exists(csvpath)

    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(csvpath)

    G_points_filepath = csvpath+"/"+str(las_fileID)+"_"+str(year)+"_GroundPoints.csv"
    GPoints_df.to_csv(G_points_filepath)

    NG_points_filepath = csvpath+"/"+str(las_fileID)+"_"+str(year)+"_NotGroundPoints.csv"
    NGPoints_df.to_csv(NG_points_filepath)

    #record end time of script
    script_end = time.time()

    print("Total Time : ",round((script_end - script_start)/60,2)," min")