import pickle

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from keras import backend as K
from keras import layers

HUBER_DELTA = 0.5


def matmul(A, B):
    return tf.matmul(A, B)


def get_model(num_points=2048):
    input_points = layers.Input(shape=(num_points, 3))
    x = layers.Convolution1D(64, 1, activation='relu', input_shape=(num_points, 3))(input_points)
    x = layers.Convolution1D(128, 1, activation='relu')(x)
    x = layers.Convolution1D(1024, 1, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=num_points)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = layers.Reshape((3, 3))(x)
    g = layers.Lambda(matmul, arguments={'B': input_T})(input_points)

    g = layers.Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = layers.Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)

    # feature transform net
    f = layers.Convolution1D(64, 1, activation='relu')(g)
    f = layers.Convolution1D(128, 1, activation='relu')(f)
    f = layers.Convolution1D(1024, 1, activation='relu')(f)
    f = layers.MaxPooling1D(pool_size=num_points)(f)
    f = layers.Dense(512, activation='relu')(f)
    f = layers.Dense(256, activation='relu')(f)
    f = layers.Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = layers.Reshape((64, 64))(f)

    # forward net
    g = layers.Lambda(matmul, arguments={'B': feature_T})(g)
    g = layers.Convolution1D(64, 1, activation='relu')(g)
    g = layers.Convolution1D(128, 1, activation='relu')(g)
    g = layers.Convolution1D(1024, 1, activation='relu')(g)

    # global_feature
    global_feature = layers.MaxPooling1D(pool_size=num_points)(g)
    global_feature = layers.Flatten()(global_feature)

    intermediate_output = np.load('intermediate_output.npy')

    resnet_activation = layers.Input(shape=(intermediate_output.shape[1],), name='intermediate_output')
    f = layers.Concatenate()([global_feature, resnet_activation])

    # Definition of MLP Layer
    f = layers.Dense(512, activation='relu')(f)
    f = layers.Dense(128, activation='relu')(f)
    f = layers.Dense(128, activation='relu')(f)
    boxes = layers.Dense(24)(f)
    classes = layers.Dense(3)(f)

    # print the model summary
    return tf.keras.Model(inputs=[input_points, resnet_activation], outputs=[boxes, classes])


def get_loss(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))

    return K.sum(x)


def train():
    points = np.load('train_points.npy')

    labels = np.load('train_labels.npy')
    labels = labels.reshape((7481, 24))
    classes = np.load('train_classes.npy')

    intermediate_output = np.load('intermediate_output.npy')
    intermediate_output = np.squeeze(intermediate_output)

    # index = np.load('permuted_indices.npy')
    index = np.random.permutation(7481)

    train_points = points[index[0:6750], :, :]
    dev_points = points[index[6750:7115], :, :]
    test_points = points[index[7115:], :, :]

    train_classes = classes[index[0:6750], :]
    dev_classes = classes[index[6750:7115], :]
    test_classes = classes[index[7115:], :]

    train_labels = labels[index[0:6750], :]
    dev_labels = labels[index[6750:7115], :]
    test_labels = labels[index[7115:], :]

    train_intermediate = intermediate_output[index[0:6750], :]
    dev_intermediate = intermediate_output[index[6750:7115], :]
    test_intermediate = intermediate_output[index[7115:], :]

    model = get_model()
    model.summary()
    # epoch number
    epochs = 500
    # compile classification model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.7),
                  loss=[get_loss, 'mean_squared_error'],
                  metrics=['accuracy'])

    history = model.fit(x=[train_points, train_intermediate], y=[train_labels, train_classes], batch_size=32,
                        epochs=epochs,
                        validation_data=([dev_points, dev_intermediate], [dev_labels, dev_classes]), shuffle=True,
                        verbose=1)

    with open('trainHistoryDict_history450', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save_weights('my_model_weights_450.h5')

    loss = model.evaluate([test_points, test_intermediate], [test_labels, test_classes], verbose=0)
    print('Test Loss:', loss)

    loss = model.evaluate([dev_points, dev_intermediate], [dev_labels, dev_classes], verbose=0)
    print('Dev Loss:', loss)


if __name__ == '__main__':
    train()
