from ftplib import FTP
import numpy as np
import laspy
import pptk

import os
from os import path

########################################################################################################
#    FTP functions
########################################################################################################

def FTP_download_lasfile(filename, datayear=2021, folderpath="Datasets/FTP_files/LiDAR/"):
    """Downlaod a las file from ftp.gis.ny.gov

    Args:
        filename (string): lasfile to download from ftp server
        datayear (_type_): which year to look at , 2017, 2021
        folderpath (_type_): where to download the file into

    Returns:
        None
    """

    assert datayear in [2017,2021], "NYC recorded lidar data only during 2017 and 2021, default is 2021"

    print("[UTIL] Function - Datayear : ",datayear)

    domain = 'ftp.gis.ny.gov'
    ftp_datadir = None
    if datayear == 2017:
        ftp_datadir =  'elevation/LIDAR/NYC_TopoBathymetric2017'
        folderpath_subdir = folderpath + "NYC_2017/"
    elif datayear == 2021:
        ftp_datadir =  'elevation/LIDAR/NYC_2021'
        folderpath_subdir = folderpath + "NYC_2021/" 
    
    #Added blocker to not redownload file if already exists
    if (path.exists(folderpath_subdir+filename)):
        print("File Exists")
    
    else:
        print("Downloading File from FTP server")

        #Login to server
        ftp = FTP(domain)  # connect to host, default port
        ftp.login()        # user anonymous, passwd anonymous@ - Loggin in as guest

        #enter data directory
        ftp.cwd(ftp_datadir)

        #download and save file to specified path
        with open(folderpath_subdir+filename, "wb") as file:
            # use FTP's RETR command to download the file
            ftp.retrbinary(f"RETR {filename}", file.write)

        #Close FTP connection
        ftp.close()

    return None

def FTP_GetFileList():


def FTP_list_files(datayear=2021):
    """List all files in the lidar directory of NYC scans

    Args:
        datayear (int, optional): _description_. Defaults to 2021.

    Returns:
        None: _description_
    """

    assert datayear in [2017,2021], "NYC recorded lidar data only during 2017 and 2021, default is 2021"

    domain = 'ftp.gis.ny.gov'
    ftp_datadir = None
    if datayear == 2017:
        ftp_datadir =  'elevation/LIDAR/NYC_TopoBathymetric2017'
    elif datayear == 2021:
        ftp_datadir =  'elevation/LIDAR/NYC_2021'
    
    #Login to server
    ftp = FTP(domain)  # connect to host, default port
    ftp.login()                     # user anonymous, passwd anonymous@

    #enter data directory
    ftp.cwd(ftp_datadir)

    ftp.retrlines('LIST')

    return None

########################################################################################################
########################################################################################################

def Write_lasFile(RawPoints, filename, pointlabels,
    Path='Datasets/Package_Generated/', #Where to store
    xscale=0.1, yscale=0.1, zscale=0.1, #lasFile default params
    xoffset=0.0, yoffset=0.0, zoffset=0.0):

    points = np.array(RawPoints)
    las = laspy.create(file_version="1.4", point_format=3)

    Xscale = xscale
    Yscale = yscale
    Zscale = zscale

    Xoffset = xoffset
    Yoffset = yoffset
    Zoffset = zoffset

    las.header.offsets = [Xoffset,Yoffset,Zoffset]
    las.header.scales = [Xscale,Yscale,Zscale]

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.intensity = [0]*len(points)
    las.classification =  pointlabels
    las.return_number =  [0]*len(points)
    las.number_of_returns =  [0]*len(points)

    las.write(Path+filename+".las")

    print("LasFile Written! Name - : ",Path+filename+".las")



def View3Dpoints_inscript(points, color=[[1,0,0]]):
    """Calls PPTK with basic config to plot 3d points

    Args:
        points (Nx3 Numpy Array): NX3 numpy array
    """
    exitViewerFlag = False
    while not exitViewerFlag:
        v = pptk.viewer(points, color*len(points))
        v.set(show_grid=False)
        v.set(show_axis=False)
        v.set(bg_color = [0,0,0,0])
        v.set(point_size = 0.0004)
        exitViewerFlag = int(input("Enter a 1 to exit viewer : "))

    v.close()

    return None

########################################################################################################
#    HyperParameter Tuning functions
########################################################################################################