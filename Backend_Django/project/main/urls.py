"""ELISA_Django_Server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from main import views
from rest_framework.documentation import include_docs_urls

urlpatterns = [
    path('', views.rest_index, name="api"),
    path('platform', views.Platform.as_view(), name="platform"),
    path('article', views.ArticleView.as_view(), name="article"),
    path('post', views.PostView.as_view(), name="post"),
    path('docs', include_docs_urls(title='ELISA API DOC')),
]
