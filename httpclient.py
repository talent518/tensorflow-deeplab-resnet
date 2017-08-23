#!/bin/env python
#-*- encoding:UTF-8 -*-

import json
import urllib2
import base64
import os

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
        response = e.read()
        # print '------------------------------'
        # print 'url: ', e.geturl()
        # print 'code: ', e.code
        # print 'reason: ', e.reason
        # print 'response: ', response
        # print '------------------------------'
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
    response, status = request('/train', {'image': base64.encodestring(readfile('/home/VOCdevkit/JPEGImages/2007_003169.jpg')), 'label': base64.encodestring(readfile('/home/VOCdevkit/SegmentationClassAug/2007_003169.png'))})

    if status:
        file = './2007_000032-mask.png'
        writefile(file, base64.decodestring(response))
        print 'Save to ' + file
    else:
        print response

    response, status = request('/test', {'image': base64.encodestring(readfile('./2007_000032.jpg'))})

    if status:
        file = './2007_000032-mask.png'
        writefile(file, base64.decodestring(response))
        print 'Save to ' + file
    else:
        print response
