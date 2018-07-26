# Very alpha version of my own face                 #
# detector program, based on work of others         #
# All rights reserved to respective owners          #
# @2018 Jacob Stanislawski                          #


import numpy as np
import cv2
import os

#Prepare the namelist for all faces we have in database - remember, the index of face must be equal to the number of folder s(number)
subjects = ["", "Kuba", "Janusz", "Obama"]

#Create the classifier to find faces later on
faceFinder = cv2.CascadeClassifier('haarcascade_frontalface.xml')

#Detecting faces in our database
def detect_face(img):
    #Convert to gray colors - required by OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Find faces
    faces = faceFinder.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    #If we cant detect face, skip it
    if (len(faces) == 0):
        return None, None
    
    #This part works only if there is only one face on source photo
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

#Prepare "food" for our program
def prepare_training_data(data_folder_path):
    #Directories to our folders with photos
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    #For each folder save the number, prepare the face-belongs-to-number data and give it to program to learn
    for dir_name in dirs:
        
        #If we find any system files, skip
        if not dir_name.startswith("s"):    
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
           
            if image_name.startswith("."):
                continue
            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    return faces, labels

faces, labels = prepare_training_data("face-database")

#This part we tell program to start learning on all the data we prepared, the "food"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

#Start capturing
cap = cv2.VideoCapture(0)

while(True):
    #Capture single frame from camera
    ret, img = cap.read()

    #Convert to gray for OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Find all faces on frame
    faces = faceFinder.detectMultiScale(gray, 1.3, 5)

    
    for (x,y,w,h) in faces:
        #Draw rectangle for each face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        #Find the index of person to whom the detected face belongs
        label, confidence = face_recognizer.predict(gray[y:y+h,x:x+w])

        #Name the person based on the array of names
        whois = subjects[label]
        name = whois + "{0:.2f}%".format(round(100 - confidence, 2))
        cv2.putText(img, str(name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #Return frame with detected faces
    cv2.imshow('Recognizer', img)
    
    #To exit press Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Shutdown the program 
cap.release()
cv2.destroyAllWindows()