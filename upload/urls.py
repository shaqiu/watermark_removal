#!/usr/bin/python3
# coding = utf-8
from django.urls import path, include
from upload import views

urlpatterns = [
    path('', views.upload),
]
