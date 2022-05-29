from openpyxl import load_workbook

def load_xls(file_name):
    try:
        wb=load_workbook(file_name,data_only=True)
        return wb
    except():
        print("load error")

def get_cell_data(file, sheetname ,cell):
    sheet = file.get_sheet_by_name(sheetname)
    return sheet[cell].value

def get_singleline_data(file, sheetname, firstcell, lastcell):
    cell = firstcell
    data = []
    while cell != lastcell:
        data.append(get_cell_data(file,sheetname,cell))
        if 65 <= ord(cell[1]) <= 90:
            if cell[1] == 'Z':
                cell = chr(ord(cell[:1]) + 1) + 'A' + cell[2:]
            else:
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
