from django.http import HttpResponse
from django.shortcuts import render, redirect

# Create your views here.
from watermark_removal import settings


def upload(request):
    pic = request.FILES['img']
    names = pic.name.split('.')
    # 获取上传图片的类型
    name = names[-1]

    save_path = "%s/upload/uploadPic.%s" % (settings.MEDIA_ROOT, name)
    with open(save_path, 'wb') as f:
        for content in pic.chunks():
            f.write(content)

    # 将图片类型保存到session，供重定向后模板加载图片
    request.session['type'] = name
    return redirect("/")
