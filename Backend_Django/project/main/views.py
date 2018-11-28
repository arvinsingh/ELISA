from django.shortcuts import render, HttpResponse, redirect
from .models import Platform_Model, Articles_Model, Post_Model
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import generics
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_204_NO_CONTENT
from .serializers import PlatformSerializer, ArticlesSerializer, PostSerializer
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView

# custom imports from jupyter notebooks (converted to python script) of Fake News Detection work
from .analysis.feature_generate import get_article_features
from .load import setup
from .prediction.prediction import *
from .click_bait.clickbait_detect import predict

# variables to be passed to prediction models for faster processing [preloading model objects]
global model, nlp
nlp = None # spacy model var
ready = False # to notify api users when django server is ready 
model = None # glove model

# load models only when needed - for development stage only
choice = input("\n\nDo you want to load rest data? (y/n): ")

# load heavy models when required
if 'y' in str(choice):
    nlp, model = setup()
    ready = True
    print("\n\nData and model has been loaded.\n")
else:
    ready = False
    print("\n\nNothing is loaded.\n")


class Platform(generics.ListAPIView):
    queryset = Platform_Model.objects.all()
    serializer = PlatformSerializer(queryset, many=True)

    def get(self, request):
        queryset = Platform_Model.objects.all()
        serializer = PlatformSerializer(queryset, many=True)
        return Response(serializer.data)


class ArticleView(generics.CreateAPIView):
    """
    1 - fetch headline and content from user and save it to database
    2 - call the analysis scripts to fetch the json objects containing charts and graphs
    3 - returns JSON object containing chart, user input, prediction and polarity score, reading standards and other graphs
    """
    queryset = Articles_Model.objects.all()
    serializer_class = ArticlesSerializer


    def post(self, request):
        try:
            serializer = ArticlesSerializer(data=request.data)
            if serializer.is_valid():
                # store data from request object
                headline = request.data.get('headline', None)
                content = request.data.get('content', None)
                data = {'user_input': {'headline':headline, 'content': content}}
                print("\n\nUser input : \n\nHeadline - ",headline,"\n\nContent - ",content)

                # check if server ready or not
                if not ready:
                    data['message'] = {'status': 'error', 'description': 'server is not ready!!!'}
                    return Response(data, status=status.HTTP_400_BAD_REQUEST)
                
                # check input length
                if len(headline)<5 or len(content)<200:
                    # show error if length of input is less
                    data['message'] = {'status': 'error', 'description': 'length of input is incorrect. headline > 3 words and content > 200 words.'}
                    return Response(data, status=status.HTTP_400_BAD_REQUEST)

                
                # cleaning data


                # feature generation
                global model
                set_glove_model(model)
                data['features'] = get_article_features(str(headline), str(content), nlp)

                # prediction
                data['prediction'] = check(str(headline), str(content)) 

                # save data to database
                serializer.save()
                data['message'] = {'status' : 'success', 'description': 'Data has been analyzed.'}

                # return json object containing analysis data
                return Response(data, status=status.HTTP_201_CREATED)
            return Response(status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            data = {'status' : 'server error', 'description': str(e)}
            return Response(data, status=status.HTTP_400_BAD_REQUEST)


class PostView(generics.CreateAPIView):
    """
    1 - fetch social media content 
    2 - perform click bait detection
    """
    queryset = Post_Model.objects.all()
    serializer_class = PostSerializer

    def post(self, request):
        try:
            serializer = PostSerializer(data=request.data)
            if serializer.is_valid():
                # store data from request object
                post = request.data.get('post', None)
                social_media = request.data.get('social_media', None)
                data = {'user_input': {'post':post, 'social media': social_media}}
                print("\n\nUser input : \n\nPost - ", post,"\n\nSocial Media - ",social_media)

                
                # check input length
                if len(post)<4:
                    # show error if length of input is less
                    data['message'] = {'status': 'error', 'description': 'length of input is incorrect. Should have minimum 4 characters'}
                    return Response(data, status=status.HTTP_400_BAD_REQUEST)

                
                # cleaning data


                # prediction
                data['prediction'] = predict(post)

                # save data to database
                serializer.save()
                data['message'] = {'status' : 'success', 'description': 'module successfully executed.'}

                # return json object containing analysis data
                return Response(data, status=status.HTTP_201_CREATED)
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            data = {'status' : 'server error', 'description': str(e)}
            return Response(data, status=status.HTTP_400_BAD_REQUEST)


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

