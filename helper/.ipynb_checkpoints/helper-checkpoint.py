"""
Membandingkan 2 list yang memiliki kesamaan nilai 0 pada index yang sama (irisan)

params :
--------
data1 : list
data2 : list

return :
--------
List index 
    List yang berisi index dari irisan kedua list
"""
def indexOfZero (data1,data2):
    return [i for i in range(len(data2)) if data1[i] == 0 or data2[i] == 0]

def indexOfNonZero(data):
    return [i for i in range(len(data)) if data[i] > 0]

"""
Memeriksa index yang bernilai 0 dari sebuah list

params :
--------
data : list of list
    Data multidimensi yang akan diperiksa

indexUser : int
    Index yang digunakan untuk mengakses data

fixUser : int
    Index yang pasti akan ditambahkan

maxIndex : int
    Maksimal index yang akan diperiksa

return :
--------
result : list
    List yang berisi index bilangan 0 dan fixIndex
"""
def checkIndexZeroOfData (*,data,indexUser,fixIndex,maxIndex):
    result = [i for i in range(len(data[indexUser])) if data[indexUser][i] == 0 and maxIndex > i]
    if not heyStack(fixIndex,result) :
        result.append(indexUser)
    return result

"""
Mencari isi dari sebuah data

params :
--------
needle : int
Isi yang akan dicari dari data

haystack : list
Data sebagai tempat pencarian

return :
--------
Boolean
    Jika true maka isi telah ditemukan dan sebaliknya
"""
def heyStack(needle,haystack) :
    for item in haystack :
        if item == needle :
            return True
    return False

"""
Transpose matrix

params :
--------
data : list of list
    List yang akan di Transpose

return :
--------
list of list
"""
def reverseMatrix(data) :
    return [[float(data[j][i]) for j in range(len(data))]for i in range(len(data[0]))]

"""
Menghapus item pada data

params :
--------
data : list
item : int

return :
--------
list
"""
def remove_items(data, item): 
    # using list comprehension to perform the task 
    return [i for i in data if i != item] 

"""
Memeriksa keberadaan index

params : 
--------
index : int
    Index yang akan diperiksa
data : list
    Data yang akan diperiksa

return :
--------
boolean
    Jika true maka keberadaan index terdeteksi dan sebaliknya

"""
def isNotExistIndex(index,data): 
    return 0 <= index < len(data)

"""
Membuat list dengan element tertentu

params :
--------
r1 : int
    Element list
r2 : int
    Panjang List

return :
---------
List 
    Berisi element r1 dan panjang list ketentuan dengan r2
int 
    Berisi element r1 jika nilai r1 sama dengan r2
"""
def createList(r1, r2) -> list|int :
    if (r1 == r2):
        return r1
    else:
        res = []
        while(r1 < r2+1 ):
            res.append(r1)
            r1 += 1
    return res