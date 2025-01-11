# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File :
# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import time

import requests

# test环境
ak = "替换成申请的AK"
sk = "替换成申请的SK"


def getmd5(data):
    return hashlib.md5(data.encode('utf-8')).hexdigest()


def hmacsha256(secret, message):
    data = message.encode('utf-8')
    return hmac.new(secret.encode('utf-8'), data, digestmod=hashlib.sha256).hexdigest()


if __name__ == '__main__':
    message = {
        "model": "替换成申请的model",
        "messages": [
            {
                "version":"api-v2",
                "created": 1683944235,
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "body": "肚子疼怎么办",

                    },
                ]
            },
        ]
    }
    md5 = getmd5(json.dumps(message))
    timestr = time.strftime("%d %b %Y %H:%M:%S GMT", time.localtime())
    authStringPrefix = "ihcloud/" + ak + "/" + timestr + "/300"
    signingKey = hmacsha256(sk, authStringPrefix)
    host = '替换成当前环境的host，线上环境是https://01bot.baidu.com'
    router = '/api/01bot/sse-gateway/stream'
    reqUrl = host + router
    canonicalRequest = '\n'.join(["POST", router, "content-md5:" + md5])
    signature = hmacsha256(signingKey, canonicalRequest)
    headers = {
        "Content-Type": "application/json",
        "X-IHU-Authorization-V2": authStringPrefix + "/" + signature
    }
    response = requests.post(reqUrl, data=json.dumps(message), headers=headers)
    print(response.text.encode('ISO-8859-1').decode('utf-8'))
