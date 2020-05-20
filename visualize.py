# Copyright (c) 2020 Intel Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Simple visualizer for images processed by ETA.
"""
import os
import time
import sys
import cv2
import json
import queue
import logging
import argparse
import math
import time
import numpy as np
from distutils.util import strtobool
import threading
from eis.config_manager import ConfigManager
from util.util import Util
from eis.env_config import EnvConfig
import eis.msgbus as mb
from util.log import configure_logging, LOG_LEVELS
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class SubscriberCallback:
    """Object for the databus callback to wrap needed state variables for the
    callback in to EIS.
    """

    def __init__(self, topicQueueDict, logger,
                 good_color=(0, 255, 0), bad_color=(0, 0, 255), dir_name=None,
                 save_image=False, labels=None):
        """Constructor

        :param frame_queue: Queue to put frames in as they become available
        :type: queue.Queue
        :param im_client: Image store client
        :type: GrpcImageStoreClient
        :param labels: (Optional) Label mapping for text to draw on the frame
        :type: dict
        :param good_color: (Optional) Tuple for RGB color to use for outlining
            a good image
        :type: tuple
        :param bad_color: (Optional) Tuple for RGB color to use for outlining a
            bad image
        :type: tuple
        """
        self.topicQueueDict = topicQueueDict
        self.logger = logger
        self.labels = labels
        self.good_color = good_color
        self.bad_color = bad_color
        self.dir_name = dir_name
        self.save_image = bool(strtobool(save_image))

        self.msg_frame_queue = queue.Queue(maxsize=15)

    def queue_publish(self, topic, frame):
        """queue_publish called after defects bounding box is drawn
        on the image. These images are published over the queue.

        :param topic: Topic the message was published on
        :type: str
        :param frame: Images with the bounding box
        :type: numpy.ndarray
        :param topicQueueDict: Dictionary to maintain multiple queues.
        :type: dict
        """
        for key in self.topicQueueDict:
            if (key == topic):
                if not self.topicQueueDict[key].full():
                    self.topicQueueDict[key].put_nowait(frame)
                    del frame
                else:
                    self.logger.warning("Dropping frames as the queue is full")

    def draw_defect(self, results, blob, topic, stream_label):
        """Identify the defects and draw boxes on the frames

        :param results: Metadata of frame received from message bus.
        :type: dict
        :param blob: Actual frame received from message bus.
        :type: bytes
        :param topic: Topic the message was published on
        :type: str
        :param results: Message received on the given topic (JSON blob)
        :type: str
        :return: Return classified results(metadata and frame)
        :rtype: dict and numpy array
        """
        height = int(results['height'])
        width = int(results['width'])
        channels = int(results['channels'])
        encoding = None

        if 'encoding_type' and 'encoding_level' in results:
            encoding = {"type": results['encoding_type'],
                        "level": results['encoding_level']}
        # Convert to Numpy array and reshape to frame
        self.logger.info('Preparing frame for visualization')
        frame = np.frombuffer(blob, dtype=np.uint8)
        if encoding is not None:
            frame = np.reshape(frame, (frame.shape))
            try:
                frame = cv2.imdecode(frame, 1)
            except cv2.error as ex:
                self.logger.error("frame: {}, exception: {}".format(frame, ex))
        else:
            self.logger.info("Encoding not enabled...")
            frame = np.reshape(frame, (height, width, channels))

        # Draw defects for Gva
        if 'gva_meta' in results:
            c = 0
            for d in results['gva_meta']:
                x1 = d['x']
                y1 = d['y']
                x2 = x1 + d['width']
                y2 = y1 + d['height']

                tl = tuple([x1, y1])
                br = tuple([x2, y2])

                # Draw bounding box
                cv2.rectangle(frame, tl, br, self.bad_color, 2)

                # Draw labels
                for l in d['tensor']:
                    if l['label_id'] is not None:
                        pos = (x1, y1 - c)
                        c += 10
                        if stream_label is not None and \
                           str(l['label_id']) in stream_label:
                            label = stream_label[str(l['label_id'])]
                            cv2.putText(frame, label, pos,
                                        cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                        self.bad_color, 2, cv2.LINE_AA)
                        else:
                            self.logger.error("Label id:{} not found".
                                              format(l['label_id']))

        # Draw defects
        if 'defects' in results:
            for d in results['defects']:
                d['tl'][0] = int(d['tl'][0])
                d['tl'][1] = int(d['tl'][1])
                d['br'][0] = int(d['br'][0])
                d['br'][1] = int(d['br'][1])

                # Get tuples for top-left and bottom-right coordinates
                tl = tuple(d['tl'])
                br = tuple(d['br'])

                # Draw bounding box
                cv2.rectangle(frame, tl, br, self.bad_color, 2)

                # Draw labels for defects if given the mapping
                if stream_label is not None:
                    # Position of the text below the bounding box
                    pos = (tl[0], br[1] + 20)

                    # The label is the "type" key of the defect, which
                    #  is converted to a string for getting from the labels
                    if str(d['type']) in stream_label:
                        label = stream_label[str(d['type'])]
                        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_DUPLEX,
                                    0.5, self.bad_color, 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, str(d['type']), pos,
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                    self.bad_color, 2, cv2.LINE_AA)

            # Draw border around frame if has defects or no defects
            if results['defects']:
                outline_color = self.bad_color
            else:
                outline_color = self.good_color

            frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                       value=outline_color)

        # Display information about frame FPS
        x = 20
        y = 20
        for res in results:
            if "Fps" in res:
                fps_str = "{} : {}".format(str(res), str(results[res]))
                self.logger.debug(fps_str)
                cv2.putText(frame, fps_str, (x, y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5,
                            self.good_color, 1, cv2.LINE_AA)
                y = y + 20

        # Display information about frame
        (dx, dy) = (20, 50)
        if 'display_info' in results:
            for d_i in results['display_info']:
                # Get priority
                priority = d_i['priority']
                info = d_i['info']
                dy = dy + 10

                #  LOW
                if priority == 0:
                    cv2.putText(frame, info, (dx, dy), cv2.FONT_HERSHEY_DUPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
                #  MEDIUM
                if priority == 1:
                    cv2.putText(frame, info, (dx, dy), cv2.FONT_HERSHEY_DUPLEX,
                                0.5, (0, 150, 170), 1, cv2.LINE_AA)
                #  HIGH
                if priority == 2:
                    cv2.putText(frame, info, (dx, dy), cv2.FONT_HERSHEY_DUPLEX,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return results, frame

    def save_images(self, topic, msg, frame):
        img_handle = msg['img_handle']
        tag = ''
        if 'defects' in msg:
            if msg['defects']:
                tag = 'bad_'
            else:
                tag = 'good_'
        imgname = tag + img_handle + ".png"
        cv2.imwrite(os.path.join(self.dir_name, imgname),
                    frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

    def callback(self, msgbus_cfg, topic):
        """Callback called when the databus has a new message.

        :param msgbus_cfg: config for the context creation in EISMessagebus
        :type: str
        :param topic: Topic the message was published on
        :type: str
        """
        self.logger.debug('Initializing message bus context')

        msgbus = mb.MsgbusContext(msgbus_cfg)

        self.logger.debug(f'Initializing subscriber for topic \'{topic}\'')
        subscriber = msgbus.new_subscriber(topic)
        stream_label = None

        for key in self.labels:
            if key == topic:
                stream_label = self.labels[key]
                break

        while True:
            metadata, blob = subscriber.recv()

            if metadata is not None and blob is not None:
                results, frame = self.draw_defect(metadata, blob, topic,
                                                  stream_label)

                if 'gva_meta' in metadata:
                    self.logger.info(f'Metadata is : {metadata}')

                if self.save_image:
                    self.save_images(topic, results, frame)

                self.queue_publish(topic, frame)


class Main(QThread):
    changePixmap = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.logger = None
        self.topicsList = None
        self.queueDict = None
        self.grid_rows = None
        self.grid_cols = None
        self.init()

    @staticmethod
    def assert_exists(path):
        """Assert given path exists.

        :param path: Path to assert
        :type: str
        """
        assert os.path.exists(path), 'Path: {} does not exist'.format(path)

    @staticmethod
    def msg_bus_subscriber(topic_config_list, queueDict, logger, jsonConfig):
        """msg_bus_subscriber is the ZeroMQ callback to
        subscribe to classified results
        """
        sc = SubscriberCallback(queueDict, logger,
                                dir_name=os.environ["IMAGE_DIR"],
                                save_image=jsonConfig["save_image"],
                                labels=jsonConfig["labels"])

        for topic_config in topic_config_list:
            topic, msgbus_cfg = topic_config

            callback_thread = threading.Thread(target=sc.callback,
                                               args=(msgbus_cfg, topic, ))
            callback_thread.start()

    @staticmethod
    def get_best_grid_size(n, max_ratio=None):
        """Gives number of rows and columns for the
        grid size.

        :param n: Number of topics
        :type: int
        :param max_ratio: Ratio of columns:rows
        :type: int
        :return: Returns number of columns and rows to be displayed.
        :rtype: int and int
        """
        cols, rows = 0, 0
        i = int(math.sqrt(n))
        while i > 0:
            if n % i == 0:
                rows = i
                cols = n // rows
                if max_ratio is not None and cols / rows > max_ratio:
                    cols, rows = Main.get_best_grid_size(n + 1, max_ratio)
                break
            i -= 1
        return cols, rows

    def init(self):
        QUEUE_SIZE = 10
        dev_mode = bool(strtobool(os.environ["DEV_MODE"]))

        app_name = os.environ["AppName"]
        conf = Util.get_crypto_dict(app_name)

        cfg_mgr = ConfigManager()
        config_client = cfg_mgr.get_config_client("etcd", conf)

        self.logger = configure_logging(os.environ['PY_LOG_LEVEL'].upper(),
                                        __name__, dev_mode)

        app_name = os.environ["AppName"]

        visualizerConfig = config_client.GetConfig("/" + app_name + "/config")
        with open('./schema.json', "rb") as infile:
            schema = infile.read()
            if (Util.validate_json(schema, visualizerConfig)) is not True:
                sys.exit(1)

        jsonConfig = json.loads(visualizerConfig)
        image_dir = os.environ["IMAGE_DIR"]

        # If user provides image_dir, create the directory if don't exists
        if image_dir:
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)

        self.topicsList = EnvConfig.get_topics_from_env("sub")

        self.queueDict = {}

        topic_config_list = []
        for topic in self.topicsList:
            publisher, topic = topic.split("/")
            topic = topic.strip()
            self.queueDict[topic] = queue.Queue(maxsize=QUEUE_SIZE)
            msgbus_cfg = EnvConfig.get_messagebus_config(topic, "sub",
                                                         publisher,
                                                         config_client,
                                                         dev_mode)

            mode_address = os.environ[topic + "_cfg"].split(",")
            mode = mode_address[0].strip()
            if (not dev_mode and mode == "zmq_tcp"):
                for key in msgbus_cfg[topic]:
                    if msgbus_cfg[topic][key] is None:
                        raise ValueError("Invalid Config")

            topic_config = (topic, msgbus_cfg)
            topic_config_list.append(topic_config)

        self.msg_bus_subscriber(topic_config_list, self.queueDict, self.logger,
                                jsonConfig)

        self.grid_cols, self.grid_rows = \
            self.get_best_grid_size(len(self.topicsList), 2)

    def run(self):
        try:
            # creating a numpy array filled with zeros (blank image)
            blankImg = np.zeros((300, 300, 3), dtype=np.uint8)
            imgList = [blankImg] * (self.grid_rows * self.grid_cols)
            self.changePixmap.emit(imgList)

            while True:
                imgList = [None] * (self.grid_cols * self.grid_rows)
                new_frames = False
                for i, key in enumerate(self.queueDict):
                    if not self.queueDict[key].empty():
                        imgList[i] = self.queueDict[key].get_nowait()
                        new_frames = True

                if new_frames:
                    self.changePixmap.emit(imgList)

        except KeyboardInterrupt:
            self.logger.info('Quitting...')
        except Exception:
            self.logger.exception('Error during execution:')
        finally:
            self.logger.exception('Destroying EIS databus context')
            sys.exit(1)


class AppWin(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = None
        self.labels = None
        self.main = None
        self.XPOS = 0
        self.YPOS = 0
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT = 600
        self.prev_t = {}
        self.curr_t = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle('EIS Visualizer App')
        self.setGeometry(self.XPOS, self.YPOS, self.WINDOW_WIDTH,
                         self.WINDOW_HEIGHT)
        self.grid = QGridLayout()
        self.grid.setSpacing(0)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSizeConstraint(QLayout.SetNoConstraint)

        self.main = Main(self)

        self.labels = []
        for i in range(self.main.grid_rows):
            self.labels.append([])
            for j in range(self.main.grid_cols):
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                self.labels[i].append(label)
                self.grid.addWidget(label, i, j)
        self.setLayout(self.grid)

        self.main.changePixmap.connect(lambda imgList: self.change_image(
                                       imgList))
        self.main.start()

    def process_img(self, img, text, width, height):

        """This helps in processing the image like adjusting the size of
        the image according to widget size and displaying the text on the
        image widget.

        :param img: Actual image which will be displayed.
        :type: numpy.ndarray
        :param text: Subscribed topic name will be the text.
        :type: string
        :param text: Subscribed topic name will be the text.
        :type: string
        :param width: width of the image that fits the ui widget.
        :type: int
        :param height: height of the image that fits the ui widget.
        :type: int
        :return: Returns the image with respect to pyQT version, i.e, Pixmap
        :rtype: QPixmap pixmap
        """

        # Resize (preserve aspec ratio)
        h, w, c = img.shape
        aspect_ratio_img = w / h
        aspect_ratio_box = width / height
        if aspect_ratio_img > aspect_ratio_box:
            new_w = width
            new_h = h * (width / w)
        else:
            new_w = w * (height / h)
            new_h = height
        img = cv2.resize(img, (int(new_w), int(new_h)))
        # Add text to image
        h, w, c = img.shape
        if text:
            cv2.putText(img, text, (5, 10), cv2.FONT_HERSHEY_PLAIN, 0.75,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if text in self.prev_t:
                self.prev_t[text] = self.curr_t[text]
                self.curr_t[text] = time.time()
                fps = 1 / (self.curr_t[text] - self.prev_t[text])
                cv2.putText(img, f"{fps:.1f} fps", (w-30, 10),
                            cv2.FONT_HERSHEY_PLAIN, 0.75,  (255, 255, 255), 1,
                            cv2.LINE_AA)
            else:
                self.prev_t[text] = time.time()
                self.curr_t[text] = time.time()
        # Use the correct format
        if c == 1:
            f = QImage.Format_Grayscale8
        else:
            f = QImage.Format_BGR888
        if c == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            c = 3
        # Convert to Pixmap
        return QPixmap.fromImage(QImage(img.data, w, h, w*c, f))

    def change_image(self, imgList):
        """Changes the image in respective widget (label window).

        :param imgList: List of images that needs to be updated in respective
         window.
        :type imgList: list
        """
        w = self.width() // self.grid.columnCount()
        h = self.height() // self.grid.rowCount()
        i = 0
        for r in range(self.grid.rowCount()):
            for c in range(self.grid.columnCount()):
                if imgList[i] is not None:
                    text = ''
                    if i < len(self.main.topicsList):
                        text = self.main.topicsList[i].strip()
                    p = self.process_img(imgList[i], text, w, h)
                    self.labels[r][c].setPixmap(p)
                i += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = AppWin()
    win.show()
    app.exec_()
    sys.exit(0)
