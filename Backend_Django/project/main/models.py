from django.db import models
from .choices import SOCIAL_MEDIA
# Create your models here.


class Platform_Model(models.Model):
    """
    three fields are readonly and can be altered from django admin panel
    """
    hadoop_url = models.CharField(max_length=100)
    jupyter_url = models.CharField(max_length=100)
    colab_url = models.CharField(max_length=100)

    def __str__(self):
        return "Platform URLs"

    def clean(self):
        """
        Throw ValidationError if you try to save more than one model instance
        """
        model = self.__class__
        if (model.objects.count() > 0 and self.id != model.objects.get().id):
            raise ValidationError("Can only create 1 instance of %s." % model.__name__)


class Articles_Model(models.Model):
    """
    database fields with headline and content for fake news detection
    """
    headline = models.CharField(max_length=30)
    content = models.CharField(max_length=1500)

    def __str__(self):
        """
        to save first five characters as object name in admin panel
        """
        return str(headline[:5])


class Post_Model(models.Model):
    """
    post model 
    """
    post = models.CharField(max_length=30)
    social_media = models.CharField(choices=SOCIAL_MEDIA, default="Facebook", max_length=20)

    def __str__(self):
        return str(post[:30])
