# Generated by Django 2.1.1 on 2018-11-04 09:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_auto_20181104_0909'),
    ]

    operations = [
        migrations.AlterField(
            model_name='platform_models',
            name='id',
            field=models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]
