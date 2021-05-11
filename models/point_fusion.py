import numpy as np
import tensorflow as tf
from keras import backend as K
import point_fusion_data_prep
import cv2
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

HUBER_DELTA = 0.5

from functools import reduce


def IoU(box0, box1):
    # box0: [x, y, z, d]
    l_a, b_a, h_a = abs(np.squeeze(box0[3:4] - box0[5:6]))
    l_b, b_b, h_b = abs(np.squeeze(box1[3:4] - box1[5:6]))
    d_a = np.array([l_a, b_a, h_a])
    d_b = np.array([l_b, b_b, h_b])

    box_a = [sum(box0[:][0]) / 8, sum(box0[:][1]) / 8, sum(box0[:][2]) / 8]
    box_b = [sum(box1[:][0]) / 8, sum(box1[:][1]) / 8, sum(box1[:][2]) / 8]

    r0 = d_a / 2

    s0 = np.squeeze(box_a - r0)
    e0 = np.squeeze(box_a + r0)
    r1 = d_b / 2
    s1 = np.squeeze(box_b - r1)
    e1 = np.squeeze(box_b + r1)
    overlap = [max(0, abs(min(e0[i], e1[i]) - max(s0[i], s1[i]))) for i in range(3)]
    intersection = reduce(lambda x, y: x * y, overlap)
    union = l_a * b_a * h_a + l_b * b_b * h_b - intersection
    return intersection / union


def velocorners_to_imagecorners(point, calib_data):
    '''converts velo points to image points
    Based on the readme file for 3D object detection, x = P2 * R0_rect * Tr_velo_to_cam * y
    R0_rect contains a 3x3 matrix which you need to extend to a 4x4 matrix by adding a 1 as the bottom-right element
    and 0's elsewhere.
    Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix in the same way
    Remember, that calib data has to extracted for the relevant file instance

    Input:
    point: 1 point in the velodyne coordinate, a list
    calib_data: calib file for that example

    Output:
    point in the 2D image coordinate as a 3 member list

    Scheme of i'''
    point.append(1)
    point = np.array(point).reshape((4, 1))

    R0_rect = calib_data["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    P2 = calib_data["P2"].reshape(3, 4)

    R = np.zeros((4, 4))
    R[:3, :3] = R0_rect
    R[3, 3] = 1

    T = np.zeros((4, 4))
    T[:3, :4] = Tr_velo_to_cam
    T[3, 3] = 1

    point_image = np.dot(P2, np.dot(R, np.dot(T, point)))
    return list(np.squeeze(point_image.reshape(1, 3)))


def draw_3D_boundingbox(box_corners, cl, calib_data, index, img_path, cl_prob, labels):
    """Draws 3D bounding boxes
    Input:
    box_corners: a list of 8 box corners, each being a list with (x,y,z)
    cl: classification of object for that example, a string of Car, Van, or a Pedestrian
    calib_data: calib_data for that example
    index: example_index

    Output: plots figure along with saving it as BB_index.eps in the current directory
    """
    box_truth = labels[index, :]
    box_truth = np.squeeze(box_truth.reshape((1, 8, 3)))
    image = cv2.imread(img_path)

    line_order = ([0, 1], [0, 3], [1, 5], [5, 7], [4, 7], [2, 5], [6, 7], [3, 6], [6, 1], [4, 3], [4, 2], [0, 2])

    type_c = {'Car': (0, 0, 255), 'Van': (0, 255, 150), 'Pedestrian': (150, 255, 255)}

    type_c_gt = {'Car': (255, 0, 0), 'Van': (255, 0, 0), 'Pedestrian': (255, 0, 0)}

    tracklet2d = []
    for i in box_corners:
        point_image = velocorners_to_imagecorners(list(i), calib_data)
        point_image = point_image / point_image[2]
        tracklet2d.append(point_image)
    tracklet2d = np.array(tracklet2d)

    tracklet2d_gt = []
    for i in box_truth:
        point_image = velocorners_to_imagecorners(list(i), calib_data)
        point_image = point_image / point_image[2]
        tracklet2d_gt.append(point_image)
    tracklet2d_gt = np.array(tracklet2d_gt)
    for k in line_order:
        cv2.line(image, (int(tracklet2d[k[0]][0]), int(tracklet2d[k[0]][1])),
                 (int(tracklet2d[k[1]][0]), int(tracklet2d[k[1]][1])), type_c[cl], 2)
        cv2.line(image, (int(tracklet2d_gt[k[0]][0]), int(tracklet2d_gt[k[0]][1])),
                 (int(tracklet2d_gt[k[1]][0]), int(tracklet2d_gt[k[1]][1])), type_c_gt[cl], 2)

    iou = "{0:.2f}".format(IoU(box_corners, box_truth))
    cl_prob = "{0:.2f}".format(cl_prob)

    # making figure
    plt.subplots(1, 1, figsize=(12, 4))
    plt.title("Image with 3D bounding box; IOU = " + str(iou) + "  Class Probability = " + str(cl_prob))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


# ********************************** CODE CHANGE STARTS HERE ***************************************************

def get_model(num_points=2048):
    """Point Net Model
        References:
            1. https://github.com/malavikabindhi/CS230-PointFusion

    """
    # input point cloud features
    input_points = tf.keras.layers.Input(shape=(num_points, 3), name='input_layer')

    global_feature = pointnet(input_points, num_points)

    visual_feature = np.load('intermediate_output.npy')

    resnet_activation = tf.keras.layers.Input(shape=(visual_feature.shape[1],), name='visual_feature')
    fusion_vector = tf.keras.layers.Concatenate()([global_feature, resnet_activation])

    boxes, classes = fusion_net(fusion_vector)

    return tf.keras.Model(inputs=[input_points, resnet_activation], outputs=[boxes, classes])


def fusion_net(fusion_vector):
    """
    Fusion Network
        References:
            1. https://github.com/malavikabindhi/CS230-PointFusion

    """
    net = tf.keras.layers.Dense(512, activation='relu', name='fusion_net_fc_1')(fusion_vector)
    net = tf.keras.layers.Dense(128, activation='relu', name='fusion_net_fc_2')(net)
    net = tf.keras.layers.Dense(128, activation='relu', name='fusion_net_fc_3')(net)
    boxes = tf.keras.layers.Dense(24, name='fusion_net_fc_boxes')(net)
    classes = tf.keras.layers.Dense(3, name='fusion_net_fc_classes')(net)
    return boxes, classes


def pointnet(input_points, num_points):
    """Point Net Model
    References:
        1. https://github.com/charlesq34/pointnet/tree/master
        2. https://github.com/malavikabindhi/CS230-PointFusion

    """

    # Point Net Input Transform
    transformed_inputs = input_tranform_net(input_points, num_points)
    point_cloud_transformed = tf.matmul(input_points, transformed_inputs)
    net = tf.keras.layers.Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu',
                                        name='pointnet_conv1d_1')(point_cloud_transformed)
    net = tf.keras.layers.Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu',
                                        name='pointnet_conv1d_2')(net)
    # feature transform net
    transformed_features = feature_transform_net(net, num_points)
    # forward net
    net_transformed = tf.matmul(net, transformed_features)
    net = tf.keras.layers.Convolution1D(64, 1, activation='relu', name='pointnet_conv1d_3')(net_transformed)
    net = tf.keras.layers.Convolution1D(128, 1, activation='relu', name='pointnet_conv1d_4')(net)
    net = tf.keras.layers.Convolution1D(1024, 1, activation='relu', name='pointnet_conv1d_5')(net)
    # global_feature
    global_feature = tf.keras.layers.MaxPooling1D(pool_size=num_points, name='pointnet_max_pool')(net)
    global_feature = tf.keras.layers.Flatten(name='pointnet_flatten')(global_feature)
    return global_feature


def feature_transform_net(net, num_points):
    """
    Point Net Feature Tranform Net
    References:
        1. https://github.com/charlesq34/pointnet/tree/master
        2. https://github.com/malavikabindhi/CS230-PointFusion
    """
    net = tf.keras.layers.Convolution1D(64, 1, activation='relu', name='feat_xform_conv1d_1')(net)
    net = tf.keras.layers.Convolution1D(128, 1, activation='relu', name='feat_xform_conv1d_2')(net)
    net = tf.keras.layers.Convolution1D(1024, 1, activation='relu', name='feat_xform_conv1d_3')(net)
    net = tf.keras.layers.MaxPooling1D(pool_size=num_points, name='feat_xform_max_pool')(net)
    net = tf.keras.layers.Dense(512, activation='relu', name='feat_xform_fc1_1')(net)
    net = tf.keras.layers.Dense(256, activation='relu', name='feat_xform_fc1_2')(net)
    net = tf.keras.layers.Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)],
                                name='feat_xform_fc1_3')(
        net)
    transform = tf.keras.layers.Reshape((64, 64), name='feat_xform_reshape')(net)
    return transform


def input_tranform_net(inputs, num_features):
    """ 
    Point Net Input Transform
    References:
        1. https://github.com/charlesq34/pointnet/tree/master
        2. https://github.com/malavikabindhi/CS230-PointFusion
     
    """
    x = tf.keras.layers.Convolution1D(64, 1, activation='relu', input_shape=(num_features, 3),
                                      name='input_transform_conv1d_1')(inputs)
    x = tf.keras.layers.Convolution1D(128, 1, activation='relu', name='input_transform_conv1d_2')(x)
    x = tf.keras.layers.Convolution1D(1024, 1, activation='relu', name='input_transform_conv1d_3')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=num_features, name='input_transform_max_pool')(x)
    x = tf.keras.layers.Dense(512, activation='relu', name='input_transform_fc_1')(x)
    x = tf.keras.layers.Dense(256, activation='relu', name='input_transform_fc_2')(x)
    x = tf.keras.layers.Dense(9, weights=[np.zeros([256, 9]),
                                          np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)],
                              name='input_transform_fc_3')(x)
    transform = tf.keras.layers.Reshape((3, 3), name='input_transform_reshape')(x)
    return transform


# ********************************** CODE CHANGE ENDS HERE ***************************************************


def get_loss(y_true, y_pred):
    abs_error = tf.abs(y_true - y_pred)
    quadratic = tf.minimum(abs_error, HUBER_DELTA)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + HUBER_DELTA * linear
    return tf.reduce_sum(losses)


def train():
    points = np.load('train_points.npy')

    labels = np.load('train_labels.npy')
    labels = labels.reshape((7481, 24))
    classes = np.load('train_classes.npy')

    visual_feature = np.load('intermediate_output.npy')
    visual_feature = np.squeeze(visual_feature)

    index = np.load('permuted_indices.npy')
    # index = np.random.permutation(7481)

    train_points = points[index[0:6750], :, :]
    val_points = points[index[6750:7115], :, :]
    test_points = points[index[7115:], :, :]

    train_classes = classes[index[0:6750], :]
    val_classes = classes[index[6750:7115], :]
    test_classes = classes[index[7115:], :]

    train_labels = labels[index[0:6750], :]
    val_labels = labels[index[6750:7115], :]
    test_labels = labels[index[7115:], :]

    train_intermediate = visual_feature[index[0:6750], :]
    val_intermediate = visual_feature[index[6750:7115], :]
    test_intermediates = visual_feature[index[7115:], :]

    # ********************************** CODE CHANGE STARTS HERE ***************************************************

    filepath = "point_fusion_models/model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model = get_model()

    # # epochs_done = 190
    epochs = 190

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.7),
                  loss=[get_loss, 'mean_squared_error'],
                  metrics=['accuracy'])

    history = model.fit(x=[train_points, train_intermediate], y=[train_labels, train_classes], batch_size=32,
                        epochs=epochs, validation_data=([val_points, val_intermediate], [val_labels, val_classes]),
                        shuffle=True, verbose=1, callbacks=callbacks_list, initial_epoch=epochs)

    loss = model.evaluate([test_points, test_intermediates], [test_labels, test_classes], verbose=0)
    print('Test Loss:', loss)

    i = 1
    continue_training = 'y'
    while continue_training != "n":
        init_epoch = epochs
        epochs = epochs + 10
        print("i:", i, "epochs:", epochs, "init_epochs:", init_epoch)
        new_model = load_model(filepath, custom_objects={get_loss: get_loss}, compile=False)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.7),
                          loss=[get_loss, 'mean_squared_error'],
                          metrics=['accuracy'])

        new_model.fit(x=[train_points, train_intermediate], y=[train_labels, train_classes], batch_size=32,
                      epochs=epochs, validation_data=([val_points, val_intermediate], [val_labels, val_classes]),
                      shuffle=True, verbose=1, callbacks=callbacks_list, initial_epoch=init_epoch)

        loss = new_model.evaluate([test_points, test_intermediates], [test_labels, test_classes], verbose=1)

        print('Test Loss:', loss)

        # ********************************** CODE CHANGE ENDS HERE ***************************************************

        _index = 3100
        test_point = points[_index:_index + 1]
        test_intermediate = visual_feature[_index:_index + 1]
        box, classes = new_model.predict([test_point, test_intermediate])
        calib_path = rf"C:\Users\faria\PycharmProjects\frustum-pointnets\dataset\KITTI\object\training\calib\003100.txt"
        test_calib = point_fusion_data_prep.read_calib_file(calib_path)
        img_path = r"C:\Users\faria\PycharmProjects\frustum-pointnets\dataset\KITTI\object\training\image_2\003100.png"
        box = np.squeeze(box.reshape((box.shape[0], 8, 3)))
        draw_3D_boundingbox(box, 'Car', test_calib, _index, img_path, np.max(classes), labels)

        continue_training = input("Train for 10 more epochs:")
        i += 1

    test_iou = []
    count = 0
    for i in index[7115:]:
        test_point = points[i:i + 1]
        test_intermediate = visual_feature[i:i + 1]
        box, classes = new_model.predict([test_point, test_intermediate])
        box = np.squeeze(box.reshape((box.shape[0], 8, 3)))
        box_truth = labels[i]
        box_truth = np.squeeze(box_truth.reshape((1, 8, 3)))
        iou = IoU(box, box_truth)

        test_iou.append(iou)
        if (iou < 0.5):
            count += 1
            print(i)
            print(iou)
    avg_iou = sum(test_iou) / len(test_iou)
    print(avg_iou)
    print(count / len(test_iou))


if __name__ == '__main__':
    train()
