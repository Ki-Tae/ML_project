dict1 = {
    'x' : 0.27,
    'y' : 3.4,
    'z' : 1.2,
    'mu' : 6.34
}
dict2 = {
    'x' : 0.27,
    'y' : 3.4,
    'z' : 1.2,
    'mu' : 6.34
}
dict3 = {
    'x' : 0.27,
    'y' : 3.4,
    'z' : 1.2,
    'mu' : 6.34
}
path = "C:\KT_project\dataset/file1.csv"
f = open(path, "w")
f.write("%f,\t%f,\t%f,\t%f\n" % (dict1['x'],dict1['y'],dict1['z'],dict1['mu']))
f.write("%f,\t%f,\t%f,\t%f\n" % (dict2['x'],dict2['y'],dict2['z'],dict2['mu']))
f.write("%f,\t%f,\t%f,\t%f\n" % (dict3['x'],dict3['y'],dict3['z'],dict3['mu']))
f.close()

