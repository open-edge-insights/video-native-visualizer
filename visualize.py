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
import cfgmgr.config_manager as cfg
import eis.msgbus as mb
from util.util import Util
from util.log import configure_logging
from util.common import Visualizer


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


def msg_bus_subscriber(topic_config_list, queue_dict, logger,
                       json_config):
    """msg_bus_subscriber is the ZeroMQ callback to
    subscribe to classified results
    """
    visualizer = Visualizer(queue_dict, logger,
                            dir_name=os.environ["IMAGE_DIR"],
                            save_image=json_config["save_image"],
                            labels=json_config["labels"],
                            draw_results=json_config["draw_results"])

    for topic_config in topic_config_list:
        topic, msgbus_cfg = topic_config

        callback_thread = threading.Thread(target=visualizer.callback,
                                           args=(msgbus_cfg, topic, ))
        callback_thread.start()


def main(args):
    """Main method.
    """
    # Initializing Etcd to set env variables
    ctx = cfg.ConfigMgr()
    num_of_subscribers = ctx.get_num_subscribers()
    dev_mode = ctx.is_dev_mode()

    logger = configure_logging(os.environ['PY_LOG_LEVEL'].upper(),
                               __name__, dev_mode)
    window_name = 'EIS Visualizer App'

    visualizer_config = ctx.get_app_config()
    # Validating config against schema
    with open('./schema.json', "rb") as infile:
        schema = infile.read()
        if not (Util.validate_json(schema,
                                   json.dumps(visualizer_config.get_dict()))):
            sys.exit(1)

    image_dir = os.environ["IMAGE_DIR"]

    # If user provides image_dir, create the directory if don't exists
    if image_dir:
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

    # Initializing required variables
    queue_dict = {}
    topic_config_list = []
    topics_list = []
    for index in range(num_of_subscribers):
        # Fetching subscriber element based on index
        sub_ctx = ctx.get_subscriber_by_index(index)
        # Fetching msgbus config of subscriber
        msgbus_cfg = sub_ctx.get_msgbus_config()
        # Fetching topics of subscriber
        topic = sub_ctx.get_topics()[0]
        # Adding topic & msgbus_config to
        # topic_config tuple
        topic_config = (topic, msgbus_cfg)
        topic_config_list.append(topic_config)
        topics_list.append(topic)
        queue_dict[topic] = queue.Queue(maxsize=10)

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
            button_dict[str(button_count)] = \
                tkinter.Button(root_win, text=topics_list[0])
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
                           visualizer_config)

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
