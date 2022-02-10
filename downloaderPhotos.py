import requests

file = open('dataset.csv', 'r')
print("provsa")
count =0

dizionario = ""

for line in file:
 count +=1
 if len(dizionario.split(',')) < 2000:
  
  if line.split(',')[0].split('"')[1] not in dizionario:
   dizionario += line.split(',')[0].split('"')[1] + ","
   

for i in range(1900):
 print(dizionario.split(',')[i])
 response = requests.get(dizionario.split(',')[i])
 
 file = open("datasetNotMe/image"+str(i)+".jpg", "wb")
 file.write(response.content)
 file.close()


file.close()