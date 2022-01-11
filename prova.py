import os 

directory_in_str = 'C:/Users/danie/Documents/P O D C A S T/SBONGO/EXTRAS/loghi'
directory = os.fsencode(directory_in_str)
result = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename = directory_in_str+"/"+filename
    print(filename)
    if filename.endswith(".jpg"):
        
        print(".jpg")
        
        result.append(filename)
            

print(result)