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

    # if firstcell[1:] != lastcell[1:]:
    #     if 65 <= ord(firstcell[1]) <= 90:
    #         if firstcell[2:] != lastcell[2:]:
    #             print("error")
    #             return data
    #     else:
    #         print("error")
    #         return data

    while cell != lastcell:
        data.append(get_cell_data(file,sheetname,cell))
        # if 65 <= ord(cell[1]) <= 90:
        #     cell = cell[0] + chr(ord(cell[1])+1) + cell[2:]
        # else:
        #     cell = chr(ord(cell[:1])+1) + cell[1:]
        if 65 <= ord(cell[1]) <= 90:
            cell = cell[0] + chr(ord(cell[1])+1) + cell[2:]
        elif cell[0]=='Z':
            cell = 'AA' + cell[1:]
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

def get_many_column(file, sheetname, firstcell, lastcell, n):
    cell = firstcell
    data = []

    if firstcell[:1] != lastcell[:1]:
        print("error")
        return data

    while cell != lastcell:
        # tmp = get_singleline_data(file, sheetname, cell, cell[:1]+str(int(cell[1:])+n+1))
        tmp = []
        for _ in range(n):
            print(chr(ord(cell[:1]) + _) + cell[1:])
            tmp.append(get_cell_data(file,sheetname, chr(ord(cell[:1]) + _) + cell[1:]))
        print(tmp)
        print( type(tmp[0]))
        if str(type(tmp[0]))!="<class 'int'>":
            if tmp[0].month >= 6 and tmp[0].month <= 10:
                data.append(tmp)
        else:
            data.append(tmp)
        cell = cell[:1]+str(int(cell[1:])+1)
    return data

def all_data_fetch(file, sheetnamelist, firstcelllist, lastcelllist, yearly_or_monthly):
    data = []
    # input_info = [('기온','B2'),('기온','C2'),('강수량','B2')]
    year_num = 33

    for y in range(year_num):
        count = 0
        for i in range(len(sheetnamelist)):
            sheetname = sheetnamelist[i]
            firstcell = firstcelllist[i]
            lastcell = lastcelllist[i]
            fcell = firstcell
            lcell = lastcell






        # while get_cell_data(file, sheetname, fcell) != None and get_cell_data(file, sheetname, fcell) != '_':
        #     # print(fcell, lcell)
        #     data.append(get_singleline_data(file, sheetname, fcell, lcell))
        #     if 65 <= ord(fcell[1]) <= 90:
        #         fcell = fcell[:2] + str(int(fcell[2:]) + 1)
        #         lcell = lcell[:2] + str(int(lcell[2:]) + 1)
        #     else:
        #         fcell = fcell[:1] + str(int(fcell[1:]) + 1)
        #         lcell = lcell[:1] + str(int(lcell[1:]) + 1)
        #     # print(fcell + ' ' + lcell)

    return data

# 테스트용
if __name__ == '__main__':
    file = load_xls('data/maple.xlsx')
    # start_date = get_single_column(file, '단풍시기', 'B2', 'B35')
    # for i in range(len(start_date)):
    #     start_date[i] = int(start_date[i].strftime("%Y%m%d"))
    # min_temper = get_single_column(file,'기온','D14','D408')
    # max_temper = get_single_column(file,'기온','E14','E408')
    # # rain_weight = get_single_column(file, '강수량', 'C2', 'C408')
    # # rain_time = get_single_column(file, '장마', 'C2', 'C35')
    # for i in range(393,-1,-1):
    #     if 0 <= i%12 < 5 or 10 <= i%12:
    #         del(min_temper[i])
    #         del(max_temper[i])
    #         # del(rain_weight[i])
    # print(len(start_date), len(max_temper)//5, len(max_temper)//5)
    # print(get_many_column(file,'기온','A2','A30',3))
    tmp = get_many_column(file, 'sheet2', 'A2', 'A414', 5)
    # print(tmp)
    # print(get_many_column(file,'sheet3','A2','A30',3))
    # print(get_many_column(file,'sheet4','A2','A30',5))
    print(np.array(tmp))
