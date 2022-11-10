from ftplib import FTP


def dummy_func():
    """Simple dummy function to check if modulew was imported correctly

    Prints "Success"

    Parameters:
        None

    Returns:
        None
    """
    print("Success")
    return None

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

    domain = 'ftp.gis.ny.gov'
    ftp_datadir = None
    if datayear == 2017:
        ftp_datadir =  'elevation/LIDAR/NYC_TopoBathymetric2017'
        folderpath_subdir = folderpath + "NYC_2017/"
    elif datayear == 2021:
        ftp_datadir =  'elevation/LIDAR/NYC_2021'
        folderpath_subdir = folderpath + "NYC_2017/"

    
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