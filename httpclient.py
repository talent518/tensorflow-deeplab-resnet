#!/bin/env python
#-*- encoding:UTF-8 -*-

import json
import urllib2
import base64
import os
import time
import argparse

def request(route, post):
    req = urllib2.Request('http://127.0.0.1:8000' + route)

    post = json.dumps(post, ensure_ascii=False, indent=4, separators=(',', ': ')).encode('utf-8')
    req.add_data(post)
    req.add_header('Content-Type', 'application/json; charset=utf-8')

    try:
        res = urllib2.urlopen(req)
        
        response = res.read()

        contentType = res.headers.getheader('content-type')
        if contentType is None:
            return response, False
        mimeType, _ = contentType.split(';')
        if mimeType != 'application/json':
            return response, False
        
        return json.loads(response), True
    except urllib2.URLError, e:
        if hasattr(e, 'read'):
            response = e.read()
        else:
            response = e.reason

        return response, False

def readfile(file):
    f = open(file, 'rb')
    fs = os.fstat(f.fileno())
    ret = f.read(fs.st_size)
    f.close()
    return ret

def writefile(file, body):
    f = open(file, 'wb')
    ret = f.write(body)
    f.close()
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deeplab Resnet Http Client.")
    parser.add_argument("--host", type=str, default='127.0.0.1', help="connect to host(default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="connect to port(default: 8000)")
    parser.add_argument("--trains", type=int, default=10, help="training steps(default: 10)")
    parser.add_argument("--train-image", type=str, help="training image")
    parser.add_argument("--train-label", type=str, help="training label")
    parser.add_argument("--test-image", type=str, help="test label(default: --train-image)")
    parser.add_argument("--test-label", type=str, help="test label(default: --test-image + suffix -mask.png)")
    args = parser.parse_args()
    if args.train_image is None:
        parser.error('Not Found arguments --train-image')
    if args.train_label is None:
        parser.error('Not Found arguments --train-label')

    response, status = request('/train', {'trains': args.trains, 'image': base64.encodestring(readfile(args.train_image)), 'label': base64.encodestring(readfile(args.train_label))})

    print '/train', status, response if not status else base64.decodestring(response)

    if args.test_image is None:
        args.test_image = args.train_image

    if args.test_label is None:
        args.test_label = args.test_image + '-mask-%d.png' % int(time.time())

    response, status = request('/test', {'image': base64.encodestring(readfile(args.test_image))})

    if status:
        writefile(args.test_label, base64.decodestring(response))
        print '/test Save to ' + args.test_label
    else:
        print '/test', response
