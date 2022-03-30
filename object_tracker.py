import time, random
import numpy as np
import csv
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# Fuse Funtion
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import warnings

from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/p2p6.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


# Arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('--input', type=str, help="Path Video Input")
# ap.add_argument('--number', type = str, help= 'Number of sample')
# args = vars(ap.parse_args())
# warnings.filterwarnings("ignore")

# writeVideo_flag = True
# video_capture = cv2.VideoCapture(args.input)
# if writeVideo_flag:
#	w = int(video_capture.get(3))
#	h = int(video_capture.get(4))
#	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#	out = cv2.VideoWriter('Sample_Video{}.avi'.format(args.number), fourcc, 15, (w, h))
#	list_file = open('detection.txt', 'w')
#	frame_index = -1

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    count = 0
    peopleOut = 0
    peopleIn = 0

    W = None
    H = None
    a_list = [0, 0]
    d_list = [0, 0]
    print(a_list)
    frame_num = 0
    while True:
        _, img = vid.read()
        #       frame = video_capture.read()
        #     if W is None or H is None:
        #        (H, W) = frame.shape[:2]

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else:
                break

        frame_num += 1

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)



        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                # bbox = track.to_tlbr()
                continue
            bbox = track.to_tlbr()
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break

            c1 = (int(bbox[0]) + int(bbox[2])) / 2
            # c2 = (int(bbox[1]) + int(bbox[3])) / 2
            c3 = int(bbox[3])
            c4 = int(bbox[1])
            # centerPoint = (int(c1), int(c2))
            buttonPoint = (int(c1), c3)
            # topPoint = (int(c1),c4)
            cv2.circle(img, buttonPoint, 4, (255, 255, 0), 1)
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)
            a_list.insert(1, str(track.track_id) + 'x:')
            a_list.insert(2, c1)
            a_list.insert(3, str(track.track_id) + 'y:')
            a_list.insert(4, c3)
            # a_list.insert(5, str(track.track_id) + 'y2:')
            # a_list.insert(6, c4)
            # Output Data
            Df = pd.DataFrame(a_list)
            Df.to_csv('P6.csv', index=False)
            # print(a_list)
            ##像素框大小計算
            Target_width = abs((int(bbox[2]) - int(bbox[0])))
            Target_height = abs((int(bbox[3]) - int(bbox[1])))
            Target_Pixel = Target_height * Target_width
            Distance = 3 * 166.5 / Target_height
            # print(str(track.track_id)+':'+str(Target_height)+'/'+str(Distance))
            print(str(track.track_id) + ',' + str(Distance) + ',' + str(c1) + ',' + str(frame_num)+','+str(c1)+','+ str(c3))   #360路徑偵測用
            #print(str(track.track_id) + ',' +str(class_name)+ ',' + str(c1) + ',' + str(c3)+','+str(frame_num))     #棒球軌跡偵測用
            #print(str(track.track_id) + ',' + str(Distance) + ',' + str(c1) + ',' + str(frame_num))
            d_list.insert(1, str(track.track_id) + ':' + str(Distance))
            Dis = pd.DataFrame(d_list)
            Dis.to_csv('Distance.csv', index=False)
            # print(str(track.track_id)+':'+str(Target_Pixel))
            # cv2.putText(img,tracker)
            # print(track.track_id, track.stateOutMetro)
        ## Route

        # cv2.line(img, (0, H // 2 + 50), (W, H // 2 + 50), (0, 0, 255), 2)
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN

        # for det in detections:
        # bbox = det.to_tlbr()
        # c1 = (int(bbox[0]) + int(bbox[2])) / 2
        # c2 = (int(bbox[1]) + int(bbox[3])) / 2
        # centerPoint = (int(c1), int(c2))
        # cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        # cv2.circle(img,centerPoint,2,(0, 0, 255), 1)

        # cv2.circle(img, (int(a_list[1]), int(a_list[2])), 2, (0, 255, 255), 2)
        """centerPoint_Save = (int(axv), int(byv))
        cv2.circle(img, centerPoint_Save, 2, (0, 255, 255), 1)"""
        # print fps on screen
        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(converted_boxes) != 0:
                for i in range(0, len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(
                        converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.ouput:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()
    ##with open('Data.csv',newline='') as csvFile:
    # writer = csv.writer(csvFile, quoting = csv.QUOTE_ALL)
    # writer.writerow(a_list)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
