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
import numpy as np
from distutils.util import strtobool
from tkinter import *
from PIL import Image, ImageTk
import threading
from libs.ConfigManager import ConfigManager
from util.util import Util
from util.msgbusutil import MsgBusUtil
import eis.msgbus as mb
from util.log import configure_logging, LOG_LEVELS

class SubscriberCallback:
    """Object for the databus callback to wrap needed state variables for the
    callback in to EIS.
    """

    def __init__(self, topicQueueDict, logger, profiling,
                 labels=None, good_color=(0, 255, 0),
                 bad_color=(0, 0, 255), dir_name=None,
                 save_image=False, display=None):
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
        self.display = display

        self.profiling = profiling
        self.logger.debug(f'Profiling is : {self.profiling}')

        self.curr_frame = 0

        self.vi_filter_input_queue_wait = 0.0
        self.e2e = 0.0

        self.timeseries_points = 0
        self.ic_pub_total = 0.0
        self.ic_pub_wait_queue = 0.0
        self.ic_to_vs = 0.0

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
                else:
                    self.logger.warning("Dropping frames as the queue is full")

    def draw_defect(self, results, blob, topic):
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

                tl = tuple([x1,y1])
                br = tuple([x2,y2])

                # Draw bounding box
                cv2.rectangle(frame, tl, br, self.bad_color, 2)

                # Draw labels
                for l in d['tensor']:
                    if l['label_id'] is not None:
                        pos = (x1, y1 - c)
                        c += 10
                        if str(l['label_id']) in self.labels:
                            label = self.labels[str(l['label_id'])]
                            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_DUPLEX,
                                    0.5, self.bad_color, 2, cv2.LINE_AA)
                        else:
                            self.logger.error("Label id:{} not found".format(l['label_id']))

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
                if self.labels is not None:
                    # Position of the text below the bounding box
                    pos = (tl[0], br[1] + 20)

                    # The label is the "type" key of the defect, which
                    #  is converted to a string for getting from the labels
                    if str(d['type']) in self.labels:
                        label = self.labels[str(d['type'])]
                        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_DUPLEX,
                                0.5, self.bad_color, 2, cv2.LINE_AA)
                    else:
                        self.logger.error("Label id:{} not found".format(d['type']))

            # Draw border around frame if has defects or no defects
            if results['defects']:
                outline_color = self.bad_color
            else:
                outline_color = self.good_color

            frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                       value=outline_color)

        # Display information about frame
        (dx, dy) = (20, 10)
        if 'display_info' in results:
            results['display_info'] = json.loads(results['display_info'])
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

        while True:
            data = subscriber.recv()

            if isinstance(data, (list, tuple, )):
                metadata, blob = data

                if self.profiling is True:
                    self.add_profile_data(metadata)

                results, frame = self.draw_defect(metadata, blob, topic)

                if 'gva_meta' in metadata:
                    self.logger.info(f'Metadata is : {metadata}')

                if self.save_image:
                    self.save_images(topic, results, frame)

                if self.display.lower() == 'true':
                    self.queue_publish(topic, frame)
                else:
                    self.logger.info(f'Classifier_results: {results}')
            else:
                if self.profiling is True:
                    self.add_profile_data_timeseries(data)
                self.logger.info(f'Classifier_results: {data}')

    @staticmethod
    def prepare_timeseries_stats(results):
        timeseries_stats = dict()
        data = results['data'].split(',')
        for d in data[1:]:
            pair = d.split('=')
            if len(pair) == 2:
                timeseries_stats[pair[0]] = pair[1].split()[0]

        diff = int(timeseries_stats['ts_idbconn_pub_exit']) - \
            int(timeseries_stats['ts_idbconn_pub_entry'])

        timeseries_stats['ic_pub_total'] = diff

        diff = int(timeseries_stats['ts_idbconn_pub_queue_exit']) - \
            int(timeseries_stats['ts_idbconn_pub_queue_entry'])

        timeseries_stats['ic_pub_wait_queue'] = diff

        diff = int(results['ts_visualize_entry']) - \
            int(timeseries_stats['ts_idbconn_pub_exit'])

        timeseries_stats['ic_to_vs'] = diff
        return timeseries_stats

    def prepare_avg_stats_timeseries(self, timeseries_stats):
        self.ic_pub_total = self.ic_pub_total + \
            timeseries_stats['ic_pub_total']

        self.ic_pub_wait_queue = self.ic_pub_wait_queue + \
            timeseries_stats['ic_pub_wait_queue']

        self.ic_to_vs = self.ic_to_vs + timeseries_stats['ic_to_vs']
        self.timeseries_points = self.timeseries_points + 1
        avg_timeseries_stats = dict()
        avg_timeseries_stats['avg_ic_pub_total'] = \
            self.ic_pub_total / self.timeseries_points

        avg_timeseries_stats['avg_ic_pub_wait_queue'] = \
            self.ic_pub_wait_queue / self.timeseries_points

        avg_timeseries_stats['avg_ic_to_vs'] = \
            self.ic_to_vs / self.timeseries_points

        return avg_timeseries_stats

    @staticmethod
    def prepare_per_frame_stats(results):
        per_frame_stats = dict()
        vi_filter_input_queue_wait = results["ts_filterQ_exit"] - results["ts_filterQ_entry"]
        vi_filter_input_queue_wait = vi_filter_input_queue_wait/1000
        e2e = results["ts_visualize_entry"] - results["ts_Ingestor_entry"]
        e2e = e2e/1000
        per_frame_stats['vi_filter_input_queue_wait'] = vi_filter_input_queue_wait
        per_frame_stats["e2e"] = e2e
        return per_frame_stats

    def prepare_avg_stats(self, per_frame_stats):
        self.curr_frame = self.curr_frame + 1
        self.vi_filter_input_queue_wait += per_frame_stats['vi_filter_input_queue_wait']
        avg_vi_filter_input_queue_wait = self.vi_filter_input_queue_wait / self.curr_frame

        
        self.e2e += per_frame_stats["e2e"]
        avg_e2e = self.e2e / self.curr_frame

        avg_stats = dict()
        avg_stats["avg_vi_filter_input_queue_wait"] = avg_vi_filter_input_queue_wait
        avg_stats["avg_e2e"] = avg_e2e
        return avg_stats

    def add_profile_data_timeseries(self, data):
        data['ts_visualize_entry'] = time.time()*1000
        self.logger.info(f'Original timeseries is: {data}')
        timeseries_stats = SubscriberCallback.prepare_timeseries_stats(data)
        avg_timeseries_stats = \
            self.prepare_avg_stats_timeseries(timeseries_stats)
        self.logger.info(f'==========STATS START==========')
        self.logger.info(f'timeseries_stats: {timeseries_stats}')
        self.logger.info(f'avg_timeseries_stats: {avg_timeseries_stats}')
        self.logger.info(f'==========STATS END==========\n')
        return data

    def add_profile_data(self, data):
        data['ts_visualize_entry'] = time.time()*1000000
        per_frame_stats = SubscriberCallback.prepare_per_frame_stats(data)
        avg_value = self.prepare_avg_stats(per_frame_stats)

        self.logger.info(f'==========STATS START==========')
        self.logger.info(f'Original data is: {data}')
        self.logger.info(f'Per frame stats in miliseconds: {per_frame_stats}')
        self.logger.info(f'frame avg stats in miliseconds: {avg_value}')
        self.logger.info(f'==========STATS END==========\n')
        return data


def parse_args():
    """Parse command line arguments.
    """
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('-f', '--fullscreen', default=False, action='store_true',
                    help='Start visualizer in fullscreen mode')
    ap.add_argument('-l', '--labels', default=None,
                    help='JSON file mapping the defect type to labels')

    return ap.parse_args()


def assert_exists(path):
    """Assert given path exists.

    :param path: Path to assert
    :type: str
    """
    assert os.path.exists(path), 'Path: {} does not exist'.format(path)


# TODO: Enlarge individual frame on respective bnutton click

# def button_click(rootWin, frames, key):
#     topRoot=Toplevel(rootWin)
#     topRoot.title(key)

#     while True:
#         frame=frames.get()

#         img=Image.fromarray(frame)
#         image= ImageTk.PhotoImage(image=img)

#         lbl=Label(topRoot,image=image)
#         lbl.grid(row=0, column=0)
#         topRoot.update()


def msg_bus_subscriber(topic_config_list, queueDict, logger, jsonConfig,
                       labels, profiling):
    """msg_bus_subscriber is the ZeroMQ callback to
    subscribe to classified results
    """
    sc = SubscriberCallback(queueDict, logger, profiling,
                            labels=labels, dir_name=os.environ["IMAGE_DIR"],
                            save_image=jsonConfig["save_image"],
                            display=jsonConfig["display"])

    for topic_config in topic_config_list:
        topic, msgbus_cfg = topic_config

        callback_thread = threading.Thread(target=sc.callback,
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
                               __name__,dev_mode)

    app_name = os.environ["AppName"]
    window_name = 'EIS Visualizer App'

    # If user provides labels, read them in
    if args.labels is not None:
        assert_exists(args.labels)
        with open(args.labels, 'r') as f:
            labels = json.load(f)
    else:
        labels = None

    visualizerConfig = config_client.GetConfig("/" + app_name + "/config")
    jsonConfig = json.loads(visualizerConfig)
    image_dir = os.environ["IMAGE_DIR"]
    profiling = bool(strtobool(os.environ["PROFILING_MODE"]))

    # If user provides image_dir, create the directory if don't exists
    if image_dir:
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

    topicsList = MsgBusUtil.get_topics_from_env("sub")

    queueDict = {}

    if not dev_mode and jsonConfig["cert_path"] is None:
        logger.error("Kindly Provide certificate directory in etcd config"
                     " when security mode is True")
        sys.exit(1)
    topic_config_list = []
    for topic in topicsList:
        publisher, topic = topic.split("/")
        topic = topic.strip()
        queueDict[topic] = queue.Queue(maxsize=10)
        msgbus_cfg = MsgBusUtil.get_messagebus_config(topic, "sub", publisher,
                                                      config_client, dev_mode)

        mode_address = os.environ[topic + "_cfg"].split(",")
        mode = mode_address[0].strip()
        if (not dev_mode and mode == "zmq_tcp"):
            for key in msgbus_cfg[topic]:
                if msgbus_cfg[topic][key] is None:
                    raise ValueError("Invalid Config")

        topic_config = (topic, msgbus_cfg)
        topic_config_list.append(topic_config)

    if jsonConfig["display"].lower() == 'true':

        try:
            rootWin = Tk()
            buttonDict = {}
            imageDict = {}

            WINDOW_WIDTH = 600
            WINDOW_HEIGHT = 600
            windowGeometry = str(WINDOW_WIDTH) + 'x' + str(WINDOW_HEIGHT)

            rootWin.geometry(windowGeometry)
            rootWin.title(window_name)

            columnValue = len(topicsList)//2
            rowValue = len(topicsList) % 2

            heightValue = int(WINDOW_HEIGHT/(rowValue+1))
            widthValue = int(WINDOW_WIDTH/(columnValue+1))

            blankImageShape = (300, 300, 3)
            blankImage = np.zeros(blankImageShape, dtype=np.uint8)

            text = 'Disconnected'
            textPosition = (20, 250)
            textFont = cv2.FONT_HERSHEY_PLAIN
            textColor = (255, 255, 255)

            cv2.putText(blankImage, text, textPosition, textFont, 2,
                        textColor, 2, cv2.LINE_AA)

            blankimg = Image.fromarray(blankImage)

            for buttonCount in range(len(topicsList)):
                buttonStr = "button{}".format(buttonCount)
                imageDict[buttonStr] = ImageTk.PhotoImage(image=blankimg)

            buttonCount, rowCount, columnCount = 0, 0, 0
            if(len(topicsList) == 1):
                heightValue = WINDOW_HEIGHT
                widthValue = WINDOW_WIDTH
                topic_text = (topicsList[0].split("/"))[1]
                buttonDict[str(buttonCount)] = Button(rootWin,
                                                      text=topic_text)
                buttonDict[str(buttonCount)].grid(sticky='NSEW')
                Grid.rowconfigure(rootWin, 0, weight=1)
                Grid.columnconfigure(rootWin, 0, weight=1)
            else:
                for key in queueDict:
                    buttonDict[str(buttonCount)] = Button(rootWin, text=key)

                    if(columnCount > columnValue):
                        rowCount = rowCount+1
                        columnCount = 0

                    if rowCount > 0:
                        heightValue = int(WINDOW_HEIGHT/(rowCount+1))
                        for key2 in buttonDict:
                            buttonDict[key2].config(height=heightValue,
                                                    width=widthValue)
                    else:
                        for key2 in buttonDict:
                            buttonDict[key2].config(height=heightValue,
                                                    width=widthValue)

                    buttonDict[str(buttonCount)].grid(row=rowCount,
                                                      column=columnCount,
                                                      sticky='NSEW')
                    Grid.rowconfigure(rootWin, rowCount, weight=1)
                    Grid.columnconfigure(rootWin, columnCount, weight=1)

                    buttonCount = buttonCount + 1
                    columnCount = columnCount + 1

            rootWin.update()
            msg_bus_subscriber(topic_config_list, queueDict, logger,
                               jsonConfig, labels, profiling)

            while True:
                buttonCount = 0
                for key in queueDict:
                    if not queueDict[key].empty():
                        frame = queueDict[key].get_nowait()
                        img = Image.fromarray(frame)
                        if len(img.split()) > 3:
                            blue, green, red, a = img.split()
                        else:
                            blue, green, red = img.split()
                        img = Image.merge("RGB", (red, green, blue))
                        imgwidth, imgheight = img.size

                        aspect_ratio = (imgwidth/imgheight) + 0.1

                        resized_width = buttonDict[
                                        str(buttonCount)].winfo_width()

                        resized_height = round(buttonDict[
                                str(buttonCount)].winfo_width()/aspect_ratio)

                        resized_img = img.resize((resized_width,
                                                  resized_height))

                        imageDict[
                            "button"+str(
                                buttonCount)] = ImageTk.PhotoImage(
                                                            image=resized_img)

                        buttonDict[str(buttonCount)].config(
                            image=imageDict["button" +
                                            str(buttonCount)],
                            compound=BOTTOM)
                        del resized_img
                    else:
                        try:
                            buttonDict[str(buttonCount)].config(
                                image=imageDict["button" +
                                                str(buttonCount)],
                                compound=BOTTOM)
                        except Exception:
                            logger.exception("Tkinter exception")
                    buttonCount = buttonCount + 1
                rootWin.update()
        except KeyboardInterrupt:
            logger.info('Quitting...')
        except Exception:
            logger.exception('Error during execution:')
        finally:
            logger.exception('Destroying EIS databus context')
            os._exit(1)
    elif jsonConfig["display"].lower() == 'false':
        msg_bus_subscriber(topic_config_list, queueDict, logger, jsonConfig,
                           labels, profiling)


if __name__ == '__main__':

    # Parse command line arguments
    args = parse_args()
    main(args)
