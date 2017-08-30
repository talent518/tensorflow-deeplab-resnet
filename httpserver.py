#!/bin/env python
#-*- encoding:UTF-8 -*-

import BaseHTTPServer
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

import httpclient

rePath = re.compile(r'[^\w]+')

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 21
NUM_STEPS = 200
LEARNING_RATE = 1e-10
WEIGHT_DECAY = 0.0005
POWER = 0.9
MOMENTUM = 0.9

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
        global input_label
        global sess
        global reduced_loss
        global train_op

        feed_dict = {input_image:base64.decodestring(image), input_label:base64.decodestring(label)}

        if not args.is_fine_tune:
            global step_ph
            global num_steps
            feed_dict[num_steps] = trains

        f = StringIO()
        for step in range(trains):
            if not args.is_fine_tune:
                feed_dict[step_ph] = step

            start_time = time.time()
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
            duration = time.time() - start_time
            msg = 'step {:d} \t loss = {:.3f}, ({:.3f} sec/step)\n'.format(step, loss_value, duration)

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
    parser.add_argument("--debug", default=False, action="store_true", help="Tensorflow's session debugger")

    parser.add_argument("--is-training", default=False, action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--is-fine-tune", default=False, action="store_true",
                        help="Using fine_tune.py in the network.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")

    return parser.parse_args()

def main():
    """Create the model and start the evaluation process."""
    global args
    global input_image
    global raw_input_image
    global input_label
    global raw_input_label
    global raw_output
    global pred
    global sess
    global loader
    global srvr
    global savedir
    global stepfile
    global step
    global reduced_loss
    global train_op

    args = get_arguments()

    input_image = tf.placeholder(tf.string)

    img = tf.image.decode_jpeg(input_image, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    raw_input_image = tf.expand_dims(img, dim=0)

    # Create network.
    net = DeepLabResNetModel({'data': raw_input_image}, is_training=args.is_training, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    if args.is_fine_tune:
        trainable = tf.trainable_variables()
    else:
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
        fc_trainable = [v for v in all_trainable if 'fc' in v.name]
        conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
        fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
        fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
        assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Label
    input_label = tf.placeholder(tf.string)
    img = tf.image.decode_png(input_label, channels=1)
    raw_input_label = tf.expand_dims(img, dim=0)

    if args.is_fine_tune:
        prediction = tf.reshape(raw_output, [-1, args.num_classes])
        label_proc = prepare_label(raw_input_label, [63, 63], num_classes=args.num_classes)
        gt = tf.reshape(label_proc, [-1, args.num_classes])

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        # Define loss and optimisation parameters.
        optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        train_op = optimiser.minimize(reduced_loss, var_list=trainable)
    else:
        global step_ph
        global num_steps

        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
        label_proc = prepare_label(raw_input_label, [63, 63], num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
        raw_gt = tf.reshape(label_proc, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)
                                                      
                                                      
        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # Define loss and optimisation parameters.
        base_lr = tf.constant(args.learning_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        num_steps = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / num_steps), args.power))
        
        opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
        opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

        grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
        grads_conv = grads[:len(conv_trainable)]
        grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
        grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

        train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
        train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
        train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

        train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

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
    # srvr = BaseHTTPServer.HTTPServer(("0.0.0.0", args.port), HTTPRequestHandler)
    
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
