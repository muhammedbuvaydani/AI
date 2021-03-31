
import base64
import json
from io import BytesIO
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.applications import inception_v3
from keras.preprocessing import image
from flask_cors import CORS
from io import StringIO
import io
#import tensorflow as tf
from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
#import face_recognition as fr
from face_util import compare_faces, face_rec , check_match

from PIL import Image
from PIL import Image, ImageOps
import matplotlib
from matplotlib import cm
# from flask_cors import CORS

app = Flask(__name__)
#CORS(app)

def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


@app.route('/detect_face/', methods=['GET', 'POST'])
def face_detect():
    resp_obj = jsonify({'success': "False"})
    req = request.get_json()
    img=request.json['img']
    print (img);
    result='False'
    json_response =face2_detect(img) 
    result=json.loads(json_response)
    print(result)
    return result

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(encoded_data.encode().decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/detect_multi_faces/', methods=['GET', 'POST'])
def face_mult_detect():
    resp_obj = jsonify({'success': "False"})
    req = request.get_json()
    img=request.json['img']
#    print (img);
    result='False'
    json_response =face_multi_detect(img) 
    result=json.loads(json_response)
    print(result)
    return result

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(encoded_data.encode().decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
@app.route('/face_match', methods=['POST','GET'])
def face_match():
    if request.method == 'POST':
        # check if the post request has the file part
        if ('file1' in request.files) and ('file2' in request.files):        
            file1 = request.files.get('file1')
            file2 = request.files.get('file2')
            ret = compare_faces(file1, file2)     
            resp_data = {"match": bool(ret)} # convert numpy._bool to bool for json.dumps
            return json.dumps(resp_data)

@app.route('/check_match', methods=['POST','GET'])
def check():
   img1=request.json['img1']
   img2=request.json['img2']   
   try:
     res= check_match(img1,img2)
     print(res)
     result=res
   except:
     result=json.dumps({"success":"false"})
   
   return result  
@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
     class_names=['id','another']
     #print(request.json['signature_name'])
#     newimage=request.json['signature_name']
#     new2=newimage
     img=request.json['img']
     print(img)
     np.set_printoptions(suppress=True)
     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
     size = (224, 224)
#     compare1="/home/muhammed/Pictures/Webcam/2021-01-21-174802.jpg"
#     compare2="/home/muhammed/Pictures/Webcam/2021-02-02-172421.jpg"
#     result  = DeepFace.verify(compare1, compare2)
#     print(result)
#####     image = Image.open('/home/muhammed/Pictures/Webcam/2021-02-02-172429.jpg')

#####     IMAGE_URL = Image.open('/home/muhammed/Pictures/Webcam/2021-02-02-172429.jpg')
#     newimage= newimage.encode("utf-8")
     
#     image = tf.image.decode_jpeg(newimage, channels=3)
#     timage= image_preprocessing(image)
#     input_ph = tf.placeholder(tf.string, shape=[None])
#     images_tensor = tf.map_fn(timage, input_ph, back_prop=False, dtype=tf.uint8)
#     images_tensor = tf.image.convert_image_dtype(images_tensor, dtype=tf.float32)
     

#####     image = ImageOps.fit(image, size, Image.ANTIALIAS)
####     image_array = np.asarray(image)
#     fh = open("imageToSave.jpg", "wb")
#     with open("imageToSave.jpg", "wb") as fh:
#           fh.write(base64.b64decode(newimage))



#     newimage= base64.b64encode(newimage)
#     input_string = newimage.decode("utf-8")
#     timage = tf.image.decode_image(newimage, channels=3)
     #timage = tf.image.resize_images(imagetensor, [150, 150])

#     imgdata = base64.b64decode(newimage)
     #image = Image.open(io.BytesIO(imgdata))
     #cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
#     image_64_decode = base64.b64decode(newimage)
#     image_result = open('baseimage.jpeg', 'wb')
#     image_result.write(image_64_decode)
#     print(image_64_decode)   
#     newimage_array = np.asarray(newimage)
#     newimagenormalized_array=newimage

##     predict_request = '{"instances" : [{"b64": "%s"}]}' %  newimage
#     newimage = ImageOps.fit(newimage, size, Image.ANTIALIAS)
#     newimage_array = np.asarray(newimage)
#(newimage.astype(np.float32) / 127.0) - 1
 #    newnormalized_image_array=np.expand_dims(newimage, 0)

#####     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
####     normalized_image_array=np.expand_dims(normalized_image_array, 0)
     # x=x.reshape
#     data = json.dumps({"signature_name": "serving_default", "instances": normalized_image_array.tolist()})

#     new_frame = Image.new("RGBA",size)
#     new_frame.paste(newimage, (0, 0), newimage.convert("RGBA"))
#     matplotlib.colors.to_rgb(newimage)
#####     newimage = np.array(newimage, dtype=np.uint8) /127.0 -1
#     newimage = np.array(newimage, dtype=np.uint8) #
###########     newnormalized_image_array=np.expand_dims(newimage, 0)
#     last_image=np.array(Image.fromarray((newimage * 255).astype(np.uint8)).resize((224, 224)))
#     last_image=cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)
    # lastimage = Image.fromarray((newimage * 255)).resize((224, 224))
    # lastimage=lastimage.array.astype(np.uint8)
    # lastimage.convert('RGB')
#     cv2.imwrite("last.png",last_image)
#     last_image = Image.fromarray(newimage)
#     XXX = tf.convert_to_tensor(newnormalized_image_array[:,:])
#     fnewimage=np.float32(newimage)[:,:,:2]

#     image_string = base64.b64decode(newnormalized_image_array)

#     last_image.save('new.png')

#     face_locations = fr.face_locations(image_string,number_of_times_to_upsample = 0, model = 'hog')    
     #predict_request = '"instances" : [{"b64": "%s"}]' % newimage
#     print(predict_request)
     lastnewnormalized_image=loadBase64Img(img)
     lastnewnormalized_image=np.array(lastnewnormalized_image, dtype=np.uint8) /127.0 -1
     lastnewnormalized_image_array=np.expand_dims(lastnewnormalized_image, 0)
     data = json.dumps({"signature_name": "serving_default","instances":lastnewnormalized_image_array.tolist() })

     #print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))
     headers = {"content-type": "application/json"}
     json_response = requests.post('http://localhost:9700/v1/models/saved_model:predict', data=data,headers=headers)
     print(json_response.text)
     predictions = json.loads(json_response.text)['predictions']
#     print(predictions)
     
#     faceimage = fr.load_image_file("last.png")
#     last=Image.fromarray(newnormalized_image_array)
#     face= fr.face_locations(faceimage, model="cnn")     
#     print(face)
     #print(np.argmax(predicti:ons[0]))
#     tt=predictions[0][0]-0.9
#     print(tt);
     pred=predictions[0][0];
     result=json.dumps({'success': "Falses","pred":pred})
     data2 = json.dumps({"img":img  })
     print(predictions[0][0])
     if np.argmax(predictions[0])==0:
        if predictions[0][0]>=0.89:
       # json_response = requests.post('http://127.0.0.1:5001/Detect', data=data2,headers=headers)
          json_response =face2_detect(img) 
          result=json.loads(json_response)
#     print(class_names[np.argmax(predictions[0])])
     #return(class_names[np.argmax(predictions[0])] )
     print(result) 
     return(result)
#     return ('yes')


def Detects_facess(img):
    resp_obj = jsonify({'success': "False"})
    req = request.get_json()


    try:
       resp_obj = functions.detects_faces(img, detector_backend='opencv', grayscale=False, enforce_detection=True)
    except:
       resp_obj = json.dumps({'success': "False"})
    #print(resp_obj)
    return resp_obj


def face2_detect(img):
    resp_obj = jsonify({'success': "False"})
    req = request.get_json()


    try:
      resp_obj = functions.detects_faces2(img, detector_backend='mtcnn', grayscale=False, enforce_detection=True)
    except:
       resp_obj = json.dumps({'success': "False"})
    return resp_obj

def face_multi_detect(img):
    resp_obj = jsonify({'success': "False"})
    req = request.get_json()


    try:
      resp_obj = functions.detects_multi(img, detector_backend='mtcnn', grayscale=False, enforce_detection=True)
    except:
       resp_obj = json.dumps({'success': "False"})
    return resp_obj
   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
