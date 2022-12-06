import argparse
import os
import time

from django.http import HttpResponse
from django.shortcuts import render, redirect

# Create your views here.
from watermark_removal import settings
import sys
sys.path.append('../')
from slbr import test_custom
from slbr import options


def upload(request):
    pic = request.FILES['img']
    names = pic.name.split('.')
    # 获取上传图片的类型
    name = names[-1]
    # 设置路径
    img_path = "%s/upload/temp" % settings.MEDIA_ROOT
    save_path = "%s/uploadPic.%s" % (img_path, name)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    with open(save_path, 'wb') as f:
        for content in pic.chunks():
            f.write(content)


    model_path = "%s/model_best.pth.tar" % settings.BASE_DIR
    res_path = "%s/result" % settings.MEDIA_ROOT
    # 设置参数，调用模型
    parser = argparse.ArgumentParser(description='WaterMark Removal')
    parser = options.Options().init(parser)
    parser.set_defaults(resume=model_path)
    parser.set_defaults(test_dir=res_path)
    print(parser.parse_args(args=[]))
    test_custom.main(img_path, name, parser.parse_args(args=[]))
    # 将图片类型保存到session，供重定向后模板加载图片
    request.session['type'] = name
    return redirect("/")

