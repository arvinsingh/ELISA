from django.shortcuts import render, HttpResponse, redirect
from .models import Platform_Model, Articles_Model, Post_Model
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import generics
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST
from .serializers import PlatformSerializer, ArticlesSerializer, PostSerializer
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView


@api_view(['GET'])
def platform(request):
    """
    to read the platform url links 
    """
    if request.method == 'GET':
        queryset = Platform_Model.objects.all()
        serializer = PlatformSerializer(queryset, many=True)
        return Response(serializer.data)


class ArticleView(generics.CreateAPIView):
    """
    1 - fetch headline and content from user and save it to database
    2 - call the analysis scripts to fetch the json objects containing charts and graphs
    """
    queryset = Articles_Model.objects.all()
    serializer_class = ArticlesSerializer

    def post(self, request):
        serializer = ArticlesSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PostView(generics.CreateAPIView):
    """
    1 - fetch social media content 
    2 - perform click bait detection
    """
    queryset = Post_Model.objects.all()
    serializer_class = PostSerializer


def index(request):
    """
        show index page while loading django server
    """
    return render(request, "index.html")

def error(request, error_url):
    """
    404 error page
    """
    return HttpResponse("<body bgcolor='black'>\
        <h3 align='center'><img src='/static/images/error404.gif'/></h3>\
        <h2 style='padding-top: 5%;color: white;' align='center'>URL <p style='color: red;'>" + str(error_url) + "</p> doesn't exist</h2>\
        </body>")
    #return redirect('/')


def rest_index(request):
    """
        custom REST Framework template
    """
    return render(request, "rest_framework/rest_index.html")
