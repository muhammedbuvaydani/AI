import face_recognition as fr
import urllib.request
import json
def check_match(file1,file2):
    url=file1
#"https://carlofontanos.com/wp-content/themes/carlo-fontanos/img/carlofontanos.jpg"
    url2=file2
#"https://carlofontanos.com/wp-content/themes/carlo-fontanos/img/carlofontanos.jpg"
    
    response = urllib.request.urlopen(url) 
    response2= urllib.request.urlopen(url2)
    image1 = fr.load_image_file(response)
    image2 = fr.load_image_file(response2)
    resp_obj = json.dumps({'success': "False"})
    try:
      image1_encoding = fr.face_encodings(image1)[0]
    except:
     resp_obj = json.dumps({'success': "False"}) 
    try:
      image2_encoding = fr.face_encodings(image2)[0]
    except:
      resp_obj = json.dumps({'success': "False"}) 
    try:  
      resp_obj2  = fr.compare_faces([image1_encoding], image2_encoding)    
    except:
      resp_obj = json.dumps({'success': "False"})
    if resp_obj2==[True]:
       resp_obj=json.dumps({'success': "True"})
    else:
       resp_obj = json.dumps({'success': "False"})
    return resp_obj


def compare_faces(file1, file2):
    """
    Compare two images and return True / False for matching.
    """
    # Load the jpg files into numpy arrays
    image1 = fr.load_image_file(file1)
    image2 = fr.load_image_file(file2)
    
    # Get the face encodings for each face in each image file
    # Assume there is only 1 face in each image, so get 1st face of an image.
    try:
     image1_encoding = fr.face_encodings(image1)[0]
    except IndexError:
     return('sorry, no 5') 
    try:
     image2_encoding = fr.face_encodings(image2)[0]
    except IndexError:
     return('sorry, no 5')    
 #   image2_encoding = fr.face_encodings(image2)[0]
    
    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = fr.compare_faces([image1_encoding], image2_encoding)    
    return results[0]

# Each face is tuple of (Name,sample image)    
known_faces = [('Obama','sample_images/obama.jpg'),
               ('Peter','sample_images/peter.jpg'),
              ]
    
def face_rec(file):
    """
    Return name for a known face, otherwise return 'Uknown'.
    """
    for name, known_file in known_faces:
        if compare_faces(known_file,file):
            return name
    return 'Unknown' 
    
def find_facial_features(file):
    # Load the jpg file into a numpy array
    image = fr.load_image_file(file)

    # Find all facial features in all the faces in the image
    face_landmarks_list = fr.face_landmarks(image)
    
    # return facial features if there is only 1 face in the image
    if len(face_landmarks_list) != 1:
        return {}
    else:
        return face_landmarks_list[0]
        
def find_face_locations(file):
    # Load the jpg file into a numpy array
    image = fr.load_image_file(file)

    # Find all face locations for the faces in the image
    face_locations = fr.face_locations(image)
    
    # return facial features if there is only 1 face in the image
    if len(face_locations) != 1:
        return []
    else:
        return face_locations[0]        
