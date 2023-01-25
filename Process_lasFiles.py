
# User Defined Package
import src.modules.utils as util
import src.modules.MultipleReturnsClassification as MRC


from multiprocessing import Pool
import logging
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score # How best can we seperate clusters
import pptk
import time

def InitiateLogger():
    logfilename = 'ProcessLas.log' 
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=logfilename, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

def Download_File(f,year,LiDAR_DatasetPath):
    #Check if file is downloaded in Dataset Directory
    util.FTP_download_lasfile(f,year,folderpath=LiDAR_DatasetPath)
    return None

def PreprocessLasFile(f, year, TileDivision, LiDAR_DatasetPath):

    #Get File_ID -> 1001.las - 1001
    las_fileID = int(''.join(c for c in f if c.isdigit())) 
    #Object to handle las preprocessing
    LasHandling = MRC.LFP
    #Get path
    lasfilepath = LiDAR_DatasetPath+'NYC_'+str(year)+'/'+str(las_fileID)+'.las'
    #Read las file
    lasfile_object = LasHandling.Read_lasFile(lasfilepath)
    #Define Tile Subdivisions
    TileDivision = TileDivision
    #rows, cols = (TileDivision, TileDivision)
    #Create Dataframe from lasfile
    lidar_df, rawpoints = LasHandling.Create_lasFileDataframe(lasfileObject=lasfile_object)

    #Extract MR and SR points from Dataframe
    MR_df = LasHandling.Get_MRpoints(lidar_df)
    SR_df = LasHandling.Get_SRpoints(lidar_df)

    return lidar_df, rawpoints, MR_df, SR_df

def Extract_GroundPoints(lidar_df):

    X_max_G , X_min_G = lidar_df.X.max(), lidar_df.X.min()
    Y_max_G , Y_min_G = lidar_df.Y.max(), lidar_df.Y.min()

    X_plane_tile_divisor_G = 60 #in m - indicates the number of tiles you want to divide the tiles into
    Y_plane_tile_divisor_G = 60 #in m

    X_diff_G = X_max_G - X_min_G
    Y_diff_G = Y_max_G - Y_min_G

    X_div_len_G = X_diff_G/X_plane_tile_divisor_G
    Y_div_len_G = Y_diff_G/Y_plane_tile_divisor_G

    # Create 2D
    rows, cols = (X_plane_tile_divisor_G, Y_plane_tile_divisor_G)
    lidar_subset_df = [[0]*cols]*rows

    Potential_Ground_Points = []
    Other_points = []

    #Iterating through each tile
    Tilecounter = 0

    for row in range(X_plane_tile_divisor_G):
        for col in range(Y_plane_tile_divisor_G):

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

    return Potential_Ground_Points_array, Potential_Other_Points_array

def Log_TileLocation(MR_df):

        #Print Lat , Long
        ix, iy = np.mean(MR_df.X.to_numpy()), np.mean(MR_df.Y.to_numpy()) 

        transformer = Transformer.from_crs("epsg:2263", "epsg:4326")
        lat, lon = transformer.transform(ix*3.28, iy*3.28)
        location_str = str(lat)+","+str(lon)

        return location_str, lat,lon

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

def ProcessLas(f,year,LiDAR_DatasetPath, EPS_distribution_global_df):

    try:

        Download_File(f,year,LiDAR_DatasetPath)

        TileDivision = 12
        lidar_df, rawpoints, MR_df, SR_df = PreprocessLasFile(f, year, TileDivision,LiDAR_DatasetPath)

        Approx_locations_str, T_lat, T_lon = Log_TileLocation(MR_df)
        logging.info("Approximate Location of %s : %s", f,Approx_locations_str)

        #lasTile class
        TileObj_SR = MRC.MR_class(SR_df,TileDivision) #Single Return Points
        TileObj_MR = MRC.MR_class(MR_df,TileDivision) #Multiple Return Points

        #Serialized Creation of Lidar Subtiles
        lidar_TilesubsetArr = TileObj_MR.Get_subtileArray()

        GPoints, NGPoints = Extract_GroundPoints(SR_df)

        # if len(GPoints) == 0:
        #     GPoints = [[0,0,0]] #default
        #     logging.warn("Emtpy Tileset found - Filename : %s", f)
        
        All_eps = [] #Stores all eps values by tile id
        N_Neighbours = 12
        subT_ID = 0

        for row in range(TileDivision):
            for col in range(TileDivision):

                if(len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

                    cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]

                    subtile_location_str, subT_lat, subT_long = Log_TileLocation(cluster_df)

                    subtile_eps = Get_eps_NN_KneeMethod(cluster_df)

                    All_eps.append(subtile_eps)

                EPS_dist_df_row = [f,T_lat,T_lon]
                EPS_dist_df_row.append(subT_ID)
                EPS_dist_df_row.append(subT_lat)
                EPS_dist_df_row.append(subT_long)
                EPS_dist_df_row.append(subtile_eps)

                EPS_distribution_global_df.loc[len(EPS_distribution_global_df.index)] = EPS_dist_df_row
                
                subT_ID = subT_ID + 1

        # fig = plt.figure(figsize=(10, 10))
        # plt.plot(All_eps)
        # plt.xlabel("Tile ID")
        # plt.ylabel("Eps value")

        Optimal_EPS = np.mean(All_eps)
        logging.info("Avg EPS for %s : %s",f,Optimal_EPS)

        EPS_CSV_filename = 'Spatial_HP_Distribution_'+f+"_"+str(year)+'.csv'
        EPS_CSV_dir = "LiDAR_HP_MATRIX/"
        logging.info("MR - T_ID : %s - ACTION: HP_MATRIX CSV file Created",f)
        EPS_distribution_global_df.to_csv(EPS_CSV_dir+EPS_CSV_filename)

        #MR

        Tilecounter = 0
        Trees_Buffer = []

        N_Neighbours = 12

        for row in range(TileDivision):
            for col in range(TileDivision):

                #print('-'*40)
                
                #print("TILE ID : ",Tilecounter)
                Tilecounter = Tilecounter + 1

                if (len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

                    cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]

                    tile_eps = Get_eps_NN_KneeMethod(cluster_df) #round(Optimal_EPS,2)
                    #print(tile_eps)

                    tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()

                    subTileTree_Points,  _ = TileObj_MR.Classify_MultipleReturns(tile_segment_points,tile_eps)

                    for t in subTileTree_Points:
                        Trees_Buffer.append(t)
                    
                    logging.info("MR - T_ID : %s - ACTION: Trees Added to - S_ID : %d",f,Tilecounter)
                
                else:
                    logging.warn("Empty Tileset Found")

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

        pptk_capture_path = LiDAR_DatasetPath + "Package_Generated/PPTK_screenshots/"
        if not os.path.exists(pptk_capture_path):
            os.makedirs(pptk_capture_path)
            logging.info("PPTK Screenshot Directory created!")

        las_fileID = int(''.join(c for c in f if c.isdigit())) 
        v.capture(pptk_capture_path+"Capture"+str(las_fileID)+"_"+str(year)+'.png')
        time.sleep(1) #screenshots were not captured after v.close was added
        v.close()

    except Exception as e:
        logging.error("Error Occured : "+str(e))
        pass


if __name__ == '__main__':

    InitiateLogger()

    # Get Year Input from User
    year = int(input("Enter Data Year to Process [2017 and 2021 supported] : "))

    # Get List of filnames on FTP server
    filenames = util.FTP_GetFileList(year)

    # Prepare arguments
    LiDAR_DatasetPath='Datasets/FTP_files/LiDAR/'
    EPS_distribution_global_df = pd.DataFrame(columns=['T_ID', 'T_lat', 'T_lon', 'subT_ID', 'subT_lat','subT_lon','EPS'])

    args = [(i, year,LiDAR_DatasetPath, EPS_distribution_global_df) for i in filenames[1200:1202]]

    #print(args)
    logging.info("LiDAR Processing Initiated")

    with Pool(2) as p:

         p.starmap(ProcessLas,args)
    

    #EPS_distribution_global_df.to_csv("Spatial_HP_Distribution.csv")



