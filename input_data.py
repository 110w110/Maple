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


def get_cell_data(file, sheetname ,cell):
    # print(sheetname)
    sheet = file.get_sheet_by_name(sheetname)
    return sheet[cell].value

def get_singleline_data(file, sheetname, firstcell, lastcell):
    cell = firstcell
    data = []

    if firstcell[1:] != lastcell[1:]:
        if 65 <= ord(firstcell[1]) <= 90:
            if firstcell[2:] != lastcell[2:]:
                print("error")
                return data
        else:
            print("error")
            return data

    while cell != lastcell:
        data.append(get_cell_data(file,sheetname,cell))
        if 65 <= ord(cell[1]) <= 90:
            cell = cell[0] + chr(ord(cell[1])+1) + cell[2:]
        else:
            cell = chr(ord(cell[:1])+1) + cell[1:]
    return data

def get_single_column(file, sheetname, firstcell, lastcell):
    cell = firstcell
    data = []

    if firstcell[:1] != lastcell[:1]:
        print("error")
        return data

    while cell != lastcell:
        data.append(get_cell_data(file,sheetname,cell))
        cell = cell[:1]+str(int(cell[1:])+1)
    return data

# 테스트용
# if __name__ == '__main__':
#     file = load_xls('temperatures.xlsx')
#     print(get_cell_data(file,'기온','A2'))
#     print(get_singleline_data(file,'기온','A2','D2'))
#     print(get_single_column(file,'기온','B2','B14'))

