#!/usr/bin/python3
# coding = utf-8

from django.http import HttpResponse
from django.shortcuts import render


def index(request):
    # 获取session保存的图片类型，如果没有则默认png类型
    type = request.session.get('type', 'png')
    return render(request, 'index.html', {'type': type})
