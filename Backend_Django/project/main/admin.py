from django.contrib import admin
from main.models import Platform_Model
# Register your models here.


class PlatformAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        if self.model.objects.count() > 0:
            return False
        else:
            return True


admin.site.register(Platform_Model, PlatformAdmin)

admin.site.index_title = "Database - Admin"

admin.site.site_header = "ELISA - Administration"

admin.site.site_title = "ELISA"