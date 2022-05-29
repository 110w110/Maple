import sys

# 정규화 함수
def normalize(arr):
    normalized = [[] for _ in range(len(arr))]
    for x in range(len(arr[0])):
        mx = -sys.maxsize
        mn = sys.maxsize
        for y in range(len(arr)):
            mx = max(mx, arr[y][x])
            mn = min(mn, arr[y][x])
        for y in range(len(arr)):
            normalized[y].append((arr[y][x] - mn) / (mx-mn))

    return normalized

# 날짜 계산 함수
def itd(num):
    month = 10
    if num > 31:
        month += 1
        num -= 31
    elif num <= 0:
        month -= 1
        num += 30
    return str(month)+"/"+str(num)