import laspy
import numpy as np
import pandas as pd

def Read_lasFile(filepath):
    """Wrapper function which runs laspy.read(lasfilepath)

    Args:
        filepath (string): Where is the .las file located

    Returns:
        las object: las object  to read
    """
    return laspy.read(filepath)

def Create_lasFileDataframe(lasfileObject):
    """Take a lasfile object after reading <filename>.las and convert into a Pandas Dataframe
    Columns Stored = {'X','Y','Z','return_number','number_of_returns'}
    Coordinates are in ft 

    Args:
        lasfileObject (_type_): lasfile Object after running Read_lasFile(filepath) function

    Returns:
        lidar_df: Pandas Datafraem of lidar points as well as columns
    """

    #Making a datframe from the lidar data
    Xscale = lasfileObject.header.x_scale
    Yscale = lasfileObject.header.y_scale
    Zscale = lasfileObject.header.z_scale

    Xoffset = lasfileObject.header.x_offset
    Yoffset = lasfileObject.header.y_offset
    Zoffset = lasfileObject.header.z_offset

    lidarPoints = np.array(
        ( 
        (lasfileObject.X*Xscale)/3.28 + Xoffset,  # convert ft to m and correct measurement
        (lasfileObject.Y*Yscale)/3.28 + Yoffset,
        (lasfileObject.Z*Zscale)/3.28 + Zoffset,
        
        lasfileObject.return_number, 
        lasfileObject.number_of_returns)).transpose()
    lidar_df = pd.DataFrame(lidarPoints , columns=['X','Y','Z','return_number','number_of_returns'])

    #Raw point cloud data
    rawPoints = np.array((
        ((lidar_df.X)*(Xscale)) + Xoffset, # convert ft to m
        (lidar_df.Y)*(Yscale) + Yoffset, #convert ft to m
        (lidar_df.Z)*(Zscale) + Zoffset
    )).transpose()

    return lidar_df, rawPoints




