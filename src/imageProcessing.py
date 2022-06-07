import utils

path = r'C:\Users\miguel\Documents\GitHub\IA-PROY03-MLP\src\data\input\images'
#img = Image.open(path)
x, y = utils.get_data_wavelet(path,0)
#Image.Image.show(img)

file_object = open(r'C:\Users\miguel\Documents\GitHub\IA-PROY03-MLP\src\data\output\imageProcessingOutput\eigenvectors.csv', 'a')
for i in x:
    for j in i:
        file_object.write(str(j) + ', ')
    file_object.write('\n')
file_object.close()