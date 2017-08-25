#!/bin/env python
#-*- encoding:UTF-8 -*-

import BaseHTTPServer
from SocketServer import ThreadingMixIn
import json
import traceback
import shutil
import re
import os
import sys
import threadpool
import time

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
from deeplab_resnet.image_reader import read_images_from_disk

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

    def action_train(self, image, label, trains=10):
        global args
        global input_image
        global raw_output
        global pred
        global sess

        image = base64.decodestring(image)
        feed_dict = {input_image:image}
    
        img = tf.image.decode_png(base64.decodestring(label), channels=1)
        input_label = tf.expand_dims(img, dim=0)
        
        prediction = tf.reshape(raw_output, [-1, args.num_classes])
        label_proc = prepare_label(input_label, tf.stack(tf.shape(input_label)[1:3]), num_classes=args.num_classes)
        gt = tf.reshape(label_proc, [-1, args.num_classes])
        print prediction.shape, gt.shape

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        # Define loss and optimisation parameters.
        optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        optim = optimiser.minimize(reduced_loss, var_list=trainable)
        f = StringIO()
        for step in range(trains):
            start_time = time.time()
            if step % 100 == 0:
                loss_value, preds, _ = sess.run([reduced_loss, pred, optim], feed_dict=feed_dict)
            else:
                loss_value, _ = sess.run([reduced_loss, optim], feed_dict=feed_dict)
            duration = time.time() - start_time
            msg = 'step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration)

            f.write(msg)
            sys.stdout.write(msg)
            sys.stdout.flush()
        length = f.tell()
        f.seek(0)
        ret = base64.encodestring(f.read(length))
        f.close()
        return ret

    def action_test(self, image):
        global args
        global input_image
        global pred
        global sess

        # Perform inference.
        image = base64.decodestring(image)
        preds = sess.run(pred, feed_dict={input_image:image})
        
        f = StringIO()
        msk = decode_labels(preds, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        im.save(f, format='PNG')
        length = f.tell()
        f.seek(0)
        ret = base64.encodestring(f.read(length))
        f.close()
        return ret

class ThreadingServer(BaseHTTPServer.HTTPServer):
    def serve_forever_thread(self, poll_interval):
        BaseHTTPServer.HTTPServer.serve_forever(self, poll_interval)

    def serve_forever(self, poll_interval=0.5):
        pool_size = cpu_count() * 2 + 1
        self.pool = threadpool.ThreadPool(pool_size)
        self.pool.putRequest(threadpool.WorkRequest(self.serve_forever_thread, args=[poll_interval]))
        try:
            while True:
                time.sleep(0.001)
                self.pool.poll()
        except KeyboardInterrupt:
            global srvr
            srvr.shutdown()
            srvr.server_close()
            srvr = None
        finally:
            print("destory all threads before exit...")
            self.pool.dismissWorkers(pool_size, do_join=True)

    def process_request_thread(self, request, client_address):
        """Same as in BaseServer but as a thread.

        In addition, exception handling is done here.

        """
        try:
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        except:
            self.handle_error(request, client_address)
            self.shutdown_request(request)

    def process_request(self, request, client_address):
        self.pool.putRequest(threadpool.WorkRequest(self.process_request_thread, args=[request, client_address]))

def cpu_count():
    '''
    Returns the number of CPUs in the system
    '''
    if sys.platform == 'win32':
        try:
            num = int(os.environ['NUMBER_OF_PROCESSORS'])
        except (ValueError, KeyError):
            num = 0
    elif 'bsd' in sys.platform or sys.platform == 'darwin':
        comm = '/sbin/sysctl -n hw.ncpu'
        if sys.platform == 'darwin':
            comm = '/usr' + comm
        try:
            with os.popen(comm) as p:
                num = int(p.read())
        except ValueError:
            num = 0
    else:
        try:
            num = os.sysconf('SC_NPROCESSORS_ONLN')
        except (ValueError, OSError, AttributeError):
            num = 0

    if num >= 1:
        return num
    else:
        return 2

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deeplab Resnet Http Server.")
    parser.add_argument("--model-weights", type=str, default=None, help="Path to the file with model weights.")
    parser.add_argument("--port", type=int, default=8000, help="Listen on port(default: 8000)")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background).")
    parser.add_argument("--debug", default=False, help="Tensorflow's session debugger")

    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")

    return parser.parse_args()

def main():
    """Create the model and start the evaluation process."""
    global args
    global input_image
    global raw_input
    global raw_output
    global pred
    global sess
    global loader
    global srvr
    global savedir
    global stepfile
    global step

    args = get_arguments()

    input_size = None # (height, width)
    input_image = tf.placeholder(tf.string)

    img = tf.image.decode_jpeg(input_image, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    raw_input = tf.expand_dims(img, dim=0)

    # Create network.
    net = DeepLabResNetModel({'data': raw_input}, is_training=args.is_training, num_classes=args.num_classes)

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    # Restore all variables, or all except the last ones.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
    trainable = [v for v in tf.trainable_variables() if 'fc1_voc12' in v.name] # Fine-tune only the last layers.
    
    # Processed predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    
    if args.debug:
        from tensorflow.python.debug import LocalCLIDebugWrapperSession as DSession
        sess = DSession(sess)

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
        pass
    except:
        traceback.print_exc()
    finally:
        print 'exiting...'
        if srvr is not None:
            srvr.shutdown()
            srvr.server_close()
        # save train result
        if sess is not None:
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            step += 1
            print 'Saving to "%s" ...' % os.path.join(savedir, 'model.ckpt-%d.*' % step)
            httpclient.writefile(stepfile, str(step))
            loader = tf.train.Saver(max_to_keep = 1000)
            loader.save(sess, os.path.join(savedir, 'model.ckpt'), global_step=step)
        print 'exited.'
