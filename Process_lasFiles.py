
# User Defined Package
import src.modules.utils as util
import src.modules.MultipleReturnsClassification as MRC


from multiprocessing import Pool
import logging

logging.basicConfig(filename='Process.log', encoding='utf-8', level=logging.DEBUG)

def ProcessLas():

    try:
        print('code here')
    except Exception as e:
        logging.error("Error Occured : ",e)


if __name__ == '__main__':

    # Get Year Input from User
    year = int(input("Enter Data Year to Process [2017 and 2021 supported] : "))
    # Get List of filnames on FTP server
    filenames = util.FTP_GetFileList(year)
    # Prepare arguments
    args = []

    logging.info('testing logging module')

    # with Pool(10) as p:

    #     p.starmap(ProcessLas,args)



