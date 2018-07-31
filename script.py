from PIL import Image
import glob
instances = []
for i in range(1,25):
    if(i<10):
        instances.append("0"+str(i))
    else:
        instances.append(str(i))
for each in instances:
    format_t = "./hpdb/"+each+"/*.png"
    for filename in glob.glob(format_t):
        img = Image.open(filename).convert('L')
        new_file = filename.split('/')[-1]
        new_file = './images/train/'+each+'/'+new_file
        img.save(new_file,'png')
