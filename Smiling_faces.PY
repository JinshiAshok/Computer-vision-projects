import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

image_path="/home/jinshi/Documents/jan_dl _bigdata/ociuz/5f3cd0410e25e-a3-5f3a7692c283d__880.jpg"
# image_path="/home/jinshi/Documents/jan_dl _bigdata/ociuz/stock-photo-dissatisfied-young-woman-keeps-arms-folded-offended-after-quarrel-with-boyfriend-ignores-live-1425770942.jpg"
# image_path = '/home/jinshi/Documents/jan_dl _bigdata/ociuz/stock-photo-collage-with-different-emotions-in-one-young-woman-ona-white-background-2139908379.jpg'
# image_path="/home/jinshi/Documents/jan_dl _bigdata/ociuz/117892636-portrait-of-her-she-his-he-nice-attractive-trendy-lovely-cheerful-people-married-spouses-pointing-at.jpg"
# image_path="/home/jinshi/Documents/jan_dl _bigdata/ociuz/Untitled.jpeg"

output_folder = '/home/jinshi/Documents/jan_dl _bigdata/ociuz/smiling_faces/'



img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)



count_smiles = 0  


os.makedirs(output_folder, exist_ok=True)

for (x, y, w, h) in faces:
    
    r_gray = gray[y:y + h, x:x + w]
    r_color = img[y:y + h, x:x + w]
    

    smiles = smile_cascade.detectMultiScale(r_gray, scaleFactor=1.8, minNeighbors=20)

    
    if len(smiles) > 0:
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(r_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            

            

           
            complete_face = r_color
            filename = f"smiling_face_{x}_{y}_{w}_{h}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), complete_face)
            print(f"Smiling face saved at: {os.path.join(output_folder, filename)}")

            count_smiles += 1  
print('no of smiles=',count_smiles)

cv2.imshow('Smile Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
