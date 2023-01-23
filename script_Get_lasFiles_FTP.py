import src.modules.utils as util # Get FTP files 
from multiprocessing import Pool # multiplrocessing

year = int(input("Which year do you want to get data from [2017 and 2021 supported] : "))

filenames = util.FTP_GetFileList(2017)

if __name__ == '__main__':
    #with Pool(10) as p:

        #p.map(util.FTP_download_lasfile,filenames)
    
    for f in filenames:
        util.FTP_download_lasfile(f)


