import tensorflow as tf
import numpy as np
import cv2
from get_features import getLbp
import os

# x_test = np.load('Data/x_test.npy')
# x_test = np.squeeze(x_test)
# lab_test = np.load('Data/lab_test.npy')
# lab_test = np.squeeze(lab_test)

# Load weights and biases of the classification model
w = np.load('Data/classification-weights.npy')
b = np.load('Data/classification-biases.npy')

# Network Parameters
n_hidden_1 = 128
n_hidden_2 = 50
n_input = 256
n_classes = 3

# tf Graph input
inp = tf.placeholder("float", [None, n_input])
outp = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(w[()]['h1']),
    'h2': tf.Variable(w[()]['h2']),
    'out': tf.Variable(w[()]['out'])
}
biases = {
    'b1': tf.Variable(b[()]['b1']),
    'b2': tf.Variable(b[()]['b2']),
    'out': tf.Variable(b[()]['out'])
}

# Create model
def multilayer_perceptron(inp, weights, biases):
    layer_1 = tf.add(tf.matmul(inp, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


pred = multilayer_perceptron(inp, weights, biases)
answer = tf.argmax(pred,1)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#####################################################################################################

# path = '/home/narshima/Documents/Face Quality/testimages'
# for fn in os.listdir(path):
#     if '.jpg' in fn:
#         l = []
#         crop = cv2.imread(path+'/'+fn)
#         f = getLbp(crop)
#         f = np.squeeze(f)
#         l.append(f)
#         p,flag = sess.run([pred,answer], feed_dict = {inp: l})
#         print fn,flag,p

#####################################################################################################

cascPath = 'Data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
    )
    for (x, y, w, h) in faces:
        w = w - 50
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop = frame[y:y+h,x:x+w]
        f = getLbp(crop)
        f = np.squeeze(f)
        l = []
        l.append(f)
        flag = sess.run(answer, feed_dict = {inp: l})
        font = cv2.FONT_HERSHEY_SIMPLEX
        if flag[0] == 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame,'GoodFace',(x, y), font, 1,(255,255,255),2,cv2.LINE_AA)
        elif flag[0] == 1:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame,'BadFace',(x, y), font, 1,(255,255,255),2,cv2.LINE_AA)
        elif flag[0] == 2:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame,'NoFace',(x, y), font, 1,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

##########################################################################################################

# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(outp, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print("Accuracy:", sess.run(accuracy, feed_dict = {inp: x_test, outp: lab_test}))
