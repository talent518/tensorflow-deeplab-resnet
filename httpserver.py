#!/bin/env python
#-*- encoding:UTF-8 -*-

import BaseHTTPServer
from SocketServer import ThreadingMixIn
import json
import traceback
import shutil
import re
import os

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from PIL import Image

import tensorflow as tf
import numpy as np
import base64
import argparse

from deeplab_resnet import DeepLabResNetModel, decode_labels, prepare_label

import httpclient

rePath = re.compile(r'[^\w]+')

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_CLASSES = 21

args = None
input_image = None
output_pred = None
sess = None
srvr = None
loader = None
step = 0
savedir = './snapshots_httpserver'
stepfile = os.path.join(savedir, 'model.ckpt.step')

class HTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def version_string(self):
        return 'Tensorflow-DeepLab-ResNet/0.1'

    def send_except(self):
        f = StringIO()
        traceback.print_exc(file=f)
        length = f.tell()
        f.seek(0)
        self.send_response(500)
        self.send_header('Content-Type', "text/plain; charset=utf-8")
        self.send_header('Content-Length', length)
        self.end_headers()
        shutil.copyfileobj(f, self.wfile)
        f.close()

    def do_POST(self):
        contentType = self.headers.getheader('content-type')
        if contentType is None:
            self.send_error(420, 'The content-type request header was not found')
            return
        mimeType, _ = contentType.split(';')
        if mimeType != 'application/json':
            self.send_error(415, 'Unsupported content-type')
            return

        contentLength = self.headers.getheader('content-length')
        if contentLength is None:
            self.send_error(421, 'The content-length request header was not found')
            return
        length = int(contentLength)
        if length <= 0:
            self.send_error(422, 'Content-Length request headers must be greater than 0')
            return
        postStr = self.rfile.read(length);
        try:
            post = json.loads(postStr)
        except:
            self.send_except()
            return

        action = 'action_' + rePath.sub('_', self.path).strip('_')
        if hasattr(self, action):
            method = getattr(self, action)
            try:
                self.post = post
                post = method(**post)
            except:
                self.send_except()
                return

        postStr = json.dumps(post, ensure_ascii=False, indent=4, separators=(',', ': ')).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Type', "application/json; charset=utf-8")
        self.send_header('Content-Length', len(postStr))
        self.end_headers()

        self.wfile.write(postStr)

    def action_train(self, image, label):
        global args
        global input_image
        global output_pred
        global sess
        global srvr

        pass

    def action_test(self, image):
        global args
        global input_image
        global output_pred
        global sess
        global srvr

        # Perform inference.
        imgfile = base64.decodestring(image)
        print type(imgfile)
        preds = sess.run(output_pred, feed_dict={input_image:imgfile})
        
        f = StringIO()
        msk = decode_labels(preds, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        im.save(f, format='PNG')
        length = f.tell()
        f.seek(0)
        ret = base64.encodestring(f.read(length))
        f.close()
        return ret

class ThreadingServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
    pass

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deeplab Resnet Http Server.")
    parser.add_argument("--model-weights", type=str, default=None, help="Path to the file with model weights.")
    parser.add_argument("--port", type=int, default=8000, help="Listen on port(default: 8000)")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background).")
    return parser.parse_args()

def main():
    """Create the model and start the evaluation process."""
    global args
    global input_image
    global output_pred
    global sess
    global loader
    global srvr
    global savedir
    global stepfile
    global step

    args = get_arguments()
    input_image = tf.placeholder(dtype=tf.string)

    # Prepare image.
    img = tf.image.decode_jpeg(input_image, channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    output_pred = tf.expand_dims(raw_output_up, dim=3)
    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    
    if os.path.exists(stepfile):
        step = int(httpclient.readfile(stepfile))
    if args.model_weights is None:
        args.model_weights = './deeplab_resnet.ckpt'
        indexfile = os.path.join(savedir, 'model.ckpt-%d.index' % step)
        if os.path.exists(indexfile):
            args.model_weights = indexfile[:-6]

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    print 'Restoring from "%s.*" ...' % args.model_weights
    loader.restore(sess, args.model_weights)

    #单线程
    # srvr = BaseHTTPServer.HTTPServer(("0.0.0.0", args.port), SimpleHTTPRequestHandler)
    
    #多线程
    srvr = ThreadingServer(("0.0.0.0", args.port), HTTPRequestHandler)

    print "serving at port", args.port
    srvr.serve_forever()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # save train result
        if loader is not None:
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            step += 1
            print 'Saving to "%s" ...' % os.path.join(savedir, 'model.ckpt-%d.*' % step)
            httpclient.writefile(stepfile, str(step))
            loader.save(sess, os.path.join(savedir, 'model.ckpt'), global_step=step)
        if srvr is not None:
            srvr.shutdown()
        print 'exited.'
