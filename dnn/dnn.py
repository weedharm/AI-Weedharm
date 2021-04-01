import numpy as np
#vẽ hình
import matplotlib.pyplot as plt
#hàm random
import random
#thư viện chữ viết tay keras
from keras.datasets import mnist
#phân rã ma trận
from keras.models import Sequential
#thư viện về chuyển các giá trị đầu vào
from keras.utils import np_utils
#hàm kích hoạt
from keras.layers.core import Dense,Activation
#tập train dùng huấn luyện, tập test để kiểm thử
#khởi tạo tập train và tập test
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#in ra thông số và giá trị ?
print("x_train shape",x_train.shape)
print("x_test shape",x_test.shape)





#chưa rõ nhưng nó ảnh hưởng đến kích thước ảnh ở cuối 
plt.rcParams['figure.figsize']=(9,9)
#tạo ra 9 ảnh và gán nhãn????? 
for i in range(9):
  plt.subplot(3,3,i+1)
  num = random.randint(0, len(x_train))
  plt.imshow(x_train[num],cmap='gray',interpolation='none')
  plt.title("Class {}".format(y_train[num]))
plt.tight_layout() # in ra


#reset về mô hình mạng nơ ron 784
x_train=x_train.reshape(60000,784)
x_train=x_train.astype('float32')
x_train/=255
x_test=x_test.reshape(10000,784)
x_test=x_test.astype('float32')
x_test/=255
print(x_train.shape)
print(x_test.shape)



#quá trình tự học dựa trên tập data
plt.tight_layout()
nb_class=10
y_train=np_utils.to_categorical(y_train,nb_class)
y_test=np_utils.to_categorical(y_test,nb_class)
model= Sequential()
model.add(Dense(10,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(20))


#test thêm chức năng
model.add(Dense(20))
model.add(Activation('relu'))


model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax')) # hàm activate
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy']) #loss funtion
model.fit(x_train,y_train,batch_size=128,epochs=5)
predicted_classes = model.predict_classes(x_test)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]



plt.figure()
for i,correct in enumerate(correct_indices[:9]):
  plt.subplot(3,3,3+i)
  plt.imshow(x_test[correct].reshape(28,28),cmap='gray',interpolation='none')
  plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
plt.tight_layout()
plt.figure()
for i,incorrect in enumerate(incorrect_indices[:9]):
  plt.subplot(3,3,3+i)
  plt.imshow(x_test[incorrect].reshape(28,28),cmap='gray',interpolation='none')
  plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
plt.tight_layout()