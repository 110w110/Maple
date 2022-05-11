from openpyxl import load_workbook
import numpy as np

def load_xls(file_name):
    try:
        wb=load_workbook(file_name,data_only=True)
        return wb
    except():
        print("load error")

def put_data_arr():
    arr=[]

    return np.array(arr)
