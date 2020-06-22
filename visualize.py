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
import sys
import json
import queue
import tkinter
import argparse
import threading
from distutils.util import strtobool
import cv2
import numpy as np
from PIL import Image, ImageTk
from eis.config_manager import ConfigManager
from eis.env_config import EnvConfig
import eis.msgbus as mb
from util.util import Util
from util.log import configure_logging


class SubscriberCallback:
    """Object for the databus callback to wrap needed state variables for the
    callback in to EIS.
    """

    def __init__(self, topic_queue_dict, logger,
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
        self.topic_queue_dict = topic_queue_dict
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
        :param topic_queue_dict: Dictionary to maintain multiple queues.
        :type: dict
        """
        for key in self.topic_queue_dict:
            if key == topic:
                if not self.topic_queue_dict[key].full():
                    self.topic_queue_dict[key].put_nowait(frame)
                    del frame
                else:
                    self.logger.warning("Dropping frames as the queue is full")

    def draw_defect(self, results, blob, stream_label):
        """Identify the defects and draw boxes on the frames

        :param results: Metadata of frame received from message bus.
        :type: dict
        :param blob: Actual frame received from message bus.
        :type: bytes
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
            self.logger.debug("Encoding not enabled...")
            frame = np.reshape(frame, (height, width, channels))

        # Draw defects for Gva
        if 'gva_meta' in results:
            count = 0
            for defect in results['gva_meta']:
                x1_axis = defect['x']
                y1_axis = defect['y']
                x2_axis = x1_axis + defect['width']
                y2_axis = y1_axis + defect['height']

                top_left = tuple([x1_axis, y1_axis])
                bottom_right = tuple([x2_axis, y2_axis])

                # Draw bounding box
                cv2.rectangle(frame, top_left, bottom_right, self.bad_color, 2)

                # Draw labels
                for label_list in defect['tensor']:
                    if label_list['label_id'] is not None:
                        pos = (x1_axis, y1_axis - count)
                        count += 10
                        if stream_label is not None and \
                           str(label_list['label_id']) in stream_label:
                            label = stream_label[str(label_lsit['label_id'])]
                            cv2.putText(frame, label, pos,
                                        cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                        self.bad_color, 2, cv2.LINE_AA)
                        else:
                            self.logger.error("Label id:{} not found".
                                              format(label_list['label_id']))

        # Draw defects
        if 'defects' in results:
            for defect in results['defects']:
                defect['tl'][0] = int(defect['tl'][0])
                defect['tl'][1] = int(defect['tl'][1])
                defect['br'][0] = int(defect['br'][0])
                defect['br'][1] = int(defect['br'][1])

                # Get tuples for top-left and bottom-right coordinates
                top_left = tuple(defect['tl'])
                bottom_right = tuple(defect['br'])

                # Draw bounding box
                cv2.rectangle(frame, top_left, bottom_right, self.bad_color, 2)

                # Draw labels for defects if given the mapping
                if stream_label is not None:
                    # Position of the text below the bounding box
                    pos = (top_left[0], bottom_right[1] + 20)

                    # The label is the "type" key of the defect, which
                    #  is converted to a string for getting from the labels
                    if str(defect['type']) in stream_label:
                        label = stream_label[str(defect['type'])]
                        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_DUPLEX,
                                    0.5, self.bad_color, 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, str(defect['type']), pos,
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
        x_axis = 20
        y_axis = 20
        for res in results:
            if "Fps" in res:
                fps_str = "{} : {}".format(str(res), str(results[res]))
                self.logger.info(fps_str)
                cv2.putText(frame, fps_str, (x_axis, y_axis),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5,
                            self.good_color, 1, cv2.LINE_AA)
                y_axis = y_axis + 20

        # Display information about frame
        (d_x, d_y) = (20, 50)
        if 'display_info' in results:
            for d_i in results['display_info']:
                # Get priority
                priority = d_i['priority']
                info = d_i['info']
                dy_axis = dy_axis + 10

                #  LOW
                if priority == 0:
                    cv2.putText(frame, info, (d_x, d_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                (0, 255, 0), 1, cv2.LINE_AA)
                #  MEDIUM
                if priority == 1:
                    cv2.putText(frame, info, (d_x, d_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                (0, 150, 170), 1, cv2.LINE_AA)
                #  HIGH
                if priority == 2:
                    cv2.putText(frame, info, (d_x, d_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)

        return results, frame

    def save_images(self, msg, frame):
        """Save_images save the image to a directory based on good or bad images.

        :param msg: metadata of the frame
        :type: str
        :param frame: Images with the bounding box
        :type: numpy.ndarray
        """
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
                results, frame = self.draw_defect(metadata, blob,
                                                  stream_label)

                self.logger.debug(f'Metadata is : {metadata}')

                if self.save_image:
                    self.save_images(results, frame)

                self.queue_publish(topic, frame)


def parse_args():
    """Parse command line arguments.
    """
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-f', '--fullscreen', default=False,
                            action='store_true',
                            help='Start visualizer in fullscreen mode')
    return arg_parser.parse_args()


def assert_exists(path):
    """Assert given path exists.

    :param path: Path to assert
    :type: str
    """
    assert os.path.exists(path), 'Path: {} does not exist'.format(path)


# TODO: Enlarge individual frame on respective bnutton click

# def button_click(root_win, frames, key):
#     topRoot=Toplevel(root_win)
#     topRoot.title(key)

#     while True:
#         frame=frames.get()

#         img=Image.fromarray(frame)
#         image= ImageTk.PhotoImage(image=img)

#         lbl=Label(topRoot,image=image)
#         lbl.grid(row=0, column=0)
#         topRoot.update()


def msg_bus_subscriber(topic_config_list, queue_dict, logger, json_config):
    """msg_bus_subscriber is the ZeroMQ callback to
    subscribe to classified results
    """
    sub_cb = SubscriberCallback(queue_dict, logger,
                                dir_name=os.environ["IMAGE_DIR"],
                                save_image=json_config["save_image"],
                                labels=json_config["labels"])

    for topic_config in topic_config_list:
        topic, msgbus_cfg = topic_config

        callback_thread = threading.Thread(target=sub_cb.callback,
                                           args=(msgbus_cfg, topic, ))
        callback_thread.start()


def main(args):
    """Main method.
    """
    dev_mode = bool(strtobool(os.environ["DEV_MODE"]))

    # Initializing Etcd to set env variables
    app_name = os.environ["AppName"]
    conf = Util.get_crypto_dict(app_name)
    cfg_mgr = ConfigManager()
    config_client = cfg_mgr.get_config_client("etcd", conf)

    logger = configure_logging(os.environ['PY_LOG_LEVEL'].upper(),
                               __name__, dev_mode)

    app_name = os.environ["AppName"]
    window_name = 'EIS Visualizer App'

    visualizer_config = config_client.GetConfig("/" + app_name + "/config")
    # Validating config against schema
    with open('./schema.json', "rb") as infile:
        schema = infile.read()
        if (Util.validate_json(schema, visualizer_config)) is not True:
            sys.exit(1)

    json_config = json.loads(visualizer_config)
    image_dir = os.environ["IMAGE_DIR"]

    # If user provides image_dir, create the directory if don't exists
    if image_dir:
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

    topics_list = EnvConfig.get_topics_from_env("sub")

    queue_dict = {}

    topic_config_list = []
    for topic in topics_list:
        publisher, topic = topic.split("/")
        topic = topic.strip()
        queue_dict[topic] = queue.Queue(maxsize=10)
        msgbus_cfg = EnvConfig.get_messagebus_config(topic, "sub", publisher,
                                                     config_client, dev_mode)

        mode_address = os.environ[topic + "_cfg"].split(",")
        mode = mode_address[0].strip()
        if (not dev_mode and mode == "zmq_tcp"):
            for key in msgbus_cfg[topic]:
                if msgbus_cfg[topic][key] is None:
                    raise ValueError("Invalid Config")

        topic_config = (topic, msgbus_cfg)
        topic_config_list.append(topic_config)

    try:
        root_win = tkinter.Tk()
        button_dict = {}
        image_dict = {}

        window_width = 600
        window_height = 600
        window_geometry = str(window_width) + 'x' + str(window_height)

        root_win.geometry(window_geometry)
        root_win.title(window_name)

        column_value = len(topics_list)//2
        row_value = len(topics_list) % 2

        height_value = int(window_height/(row_value+1))
        width_value = int(window_width/(column_value+1))

        blank_image_shape = (300, 300, 3)
        blank_image = np.zeros(blank_image_shape, dtype=np.uint8)

        text = 'Disconnected'
        text_position = (20, 250)
        text_font = cv2.FONT_HERSHEY_PLAIN
        text_color = (255, 255, 255)

        cv2.putText(blank_image, text, text_position, text_font, 2,
                    text_color, 2, cv2.LINE_AA)

        blankimg = Image.fromarray(blank_image)

        for button_count in range(len(topics_list)):
            button_str = "button{}".format(button_count)
            image_dict[button_str] = ImageTk.PhotoImage(image=blankimg)

        button_count, row_count, column_count = 0, 0, 0
        if len(topics_list) == 1:
            height_value = window_height
            width_value = window_width
            topic_text = (topics_list[0].split("/"))[1]
            button_dict[str(button_count)] = tkinter.Button(root_win,
                                                            text=topic_text)
            button_dict[str(button_count)].grid(sticky='NSEW')
            tkinter.Grid.rowconfigure(root_win, 0, weight=1)
            tkinter.Grid.columnconfigure(root_win, 0, weight=1)
        else:
            for key in queue_dict:
                button_dict[str(button_count)] = tkinter.Button(root_win,
                                                                text=key)

                if column_count > column_value:
                    row_count = row_count+1
                    column_count = 0

                if row_count > 0:
                    height_value = int(window_height/(row_count+1))
                    for key2 in button_dict:
                        button_dict[key2].config(height=height_value,
                                                 width=width_value)
                else:
                    for key2 in button_dict:
                        button_dict[key2].config(height=height_value,
                                                 width=width_value)

                button_dict[str(button_count)].grid(row=row_count,
                                                    column=column_count,
                                                    sticky='NSEW')
                tkinter.Grid.rowconfigure(root_win, row_count, weight=1)
                tkinter.Grid.columnconfigure(root_win, column_count, weight=1)

                button_count = button_count + 1
                column_count = column_count + 1

        root_win.update()
        msg_bus_subscriber(topic_config_list, queue_dict, logger,
                           json_config)

        while True:
            button_count = 0
            for key in queue_dict:
                if not queue_dict[key].empty():
                    frame = queue_dict[key].get_nowait()
                    img = Image.fromarray(frame)
                    del frame
                    if len(img.split()) > 3:
                        blue, green, red, _ = img.split()
                    else:
                        blue, green, red = img.split()
                    img = Image.merge("RGB", (red, green, blue))
                    imgwidth, imgheight = img.size

                    aspect_ratio = (imgwidth/imgheight) + 0.1

                    resized_width = button_dict[
                        str(button_count)].winfo_width()

                    resized_height = round(button_dict[
                        str(button_count)].winfo_width()/aspect_ratio)

                    resized_img = img.resize((resized_width,
                                              resized_height))
                    del img

                    image_dict[
                        "button"+str(
                            button_count)] = ImageTk.PhotoImage(
                                image=resized_img)

                    button_dict[str(button_count)].config(
                        image=image_dict["button" +
                                         str(button_count)],
                        compound=tkinter.BOTTOM)

                    del resized_img
                else:
                    try:
                        button_dict[str(button_count)].config(
                            image=image_dict["button" +
                                             str(button_count)],
                            compound=tkinter.BOTTOM)
                    except Exception:
                        logger.exception("Tkinter exception")
                button_count = button_count + 1
            root_win.update()
    except KeyboardInterrupt:
        logger.info('Quitting...')
    except Exception:
        logger.exception('Error during execution:')
    finally:
        logger.exception('Destroying EIS databus context')
        sys.exit(1)


if __name__ == '__main__':

    # Parse command line arguments
    ARGS = parse_args()
    main(ARGS)
