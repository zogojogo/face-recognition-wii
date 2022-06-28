import os

for i in range(60):
    os.system("mkdir " + str(i))

for file in os.listdir("./foto_enrollment"):
    # print(file.split(".")[0])
    os.system("cp ./foto_enrollment/" + file + " " + file.split(".")[0] + "/" + file.split(".")[0] + "_1.jpg")