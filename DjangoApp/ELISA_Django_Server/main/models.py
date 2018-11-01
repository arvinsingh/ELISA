from django.db import models

# Create your models here.


class Settings(models.Model):
    id = models.CharField(max_length=10, primary_key=True)
    hadoop_url = models.CharField(max_length=50)
    jupyter_url = models.CharField(max_length=50)
