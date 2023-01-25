
# User Defined Package
import src.modules.utils as util
import src.modules.MultipleReturnsClassification as MRC


from multiprocessing import Pool
import logging
import os

def InitiateLogger():
    logfilename = 'ProcessLas.log' 
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=logfilename, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)


def ProcessLas():

    try:
        f = open("demofile.txt")
        f.write('test')
    except Exception as e:
        logging.error("Error Occured : "+str(e))
        pass


if __name__ == '__main__':

    InitiateLogger()

    # Get Year Input from User
    year = 2017 #int(input("Enter Data Year to Process [2017 and 2021 supported] : "))
    # Get List of filnames on FTP server
    filenames = util.FTP_GetFileList(year)
    # Prepare arguments
    args = []

    logging.info('testing logging module')
    ProcessLas()

    logging.info('testing logging module line 2')

    # with Pool(10) as p:

    #     p.starmap(ProcessLas,args)



