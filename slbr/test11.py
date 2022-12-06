import argparse
import os
import time

from django.conf import settings

from options import Options
from test_custom import main

if __name__ == "__main__":

    # a = argparse.ArgumentParser(description='WaterMark Removal')
    # parser = Options().init(a)
    # img_dir = "D:\code\homework\watermark_removal\static/media/upload/temp"
    # model_path = "D:\code\homework\watermark_removal/model_best.pth.tar"
    # res_path = "D:\code\homework\watermark_removal\static/media/result"
    # parser.set_defaults(resume=model_path)
    # parser.set_defaults(test_dir=res_path)
    # main(img_dir, "jpg", parser.parse_args(args=[]))
    #
    img_path = "D:\code\homework\watermark_removal\static/media/upload/temp"
    if not os.path.exists(img_path):
        print(img_path)
        os.makedirs(img_path)
