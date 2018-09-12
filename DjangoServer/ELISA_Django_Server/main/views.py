from django.shortcuts import render
from .models import Settings
# Create your views here.


def index(request):
    data = Settings.objects.get(id="default")
    return render(request, "index.html", {'hadoop_url': data.hadoop_url})


def settings(request):
    if request.POST:
        data = Settings.objects.get(id="default")
        data.hadoop_url = request.POST['hadoop_url']
        data.save()
        return render(request, "settings.html", {'hadoop_url': data.hadoop_url})
    data = Settings.objects.get(id="default")
    return render(request, "settings.html", {'hadoop_url': data.hadoop_url})


def dataset(request):
    data = Settings.objects.get(id="default")
    return render(request, "dataset.html", {'hadoop_url': data.hadoop_url})
