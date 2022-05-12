from openpyxl import load_workbook
import numpy as np
import datetime as dt

def load_xls(file_name):
    try:
        wb=load_workbook(file_name,data_only=True)
        return wb
    except():
        print("load error")

# type 1:temperatures
def put_data_arr(wb,type):
    arr=[[]*12]
    ws=wb.active
    for row in ws.rows:
        #엑셀 형식이 다름
        month=0
        if 1<=type<=3:
            m,y=str(row[0]).split('.')
            month=month_name_to_int(m)

        # 월별로 12번 반복
        if type==1:
            arr[month-1].append([row[2],row[3]])

    return np.array(arr)

# Jan -> 1
def month_name_to_int(month_str):
    datetime_object=dt.datetime.strptime(month_str,"%b")
    month = datetime_object.month

    return month

# 88 -> 1988
def full_year(year):
    if int(year)>=88:
        year="19"+year
    else:
        year="20"+year

    return int(year)

