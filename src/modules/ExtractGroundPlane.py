import numpy as np

#Ground Plane Classifcation Algorithm
def Extract_GroundPointsAlgo(lidarSubtilePoints, ground_H_thresh_perc = 0.1):

    tile_segment_points = lidarSubtilePoints

    #Ground Seperation Algorithm

    #Get Z values of all points
    z_values = tile_segment_points[:,2]
    
    if (len(z_values) != 0):
        z_min = np.min(z_values)
        
        #Set Height Threshold for points to consider
        Z_Height_thresh = z_min*ground_H_thresh_perc

        #Seperate Ground Points and Non Ground Points
        lowest_points_idx = [idx for idx,record in enumerate(z_values) if record > (z_min) and record < (z_min+Z_Height_thresh) ]
        Ground_Points = tile_segment_points[lowest_points_idx]

        Other_points_idx =  [idx for idx,record in enumerate(z_values) if record > (z_min+Z_Height_thresh) ]
        Not_ground_points = tile_segment_points[Other_points_idx]
    
    return Ground_Points, Not_ground_points
            