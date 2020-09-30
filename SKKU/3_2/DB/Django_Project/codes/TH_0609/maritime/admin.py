from django.contrib import admin
from .models import *

admin.site.register(Comp)
admin.site.register(Mate_temp)
admin.site.register(Mate_stored)
admin.site.register(Mate_rejected)
admin.site.register(Eng_temp)
admin.site.register(Eng_stored)
admin.site.register(Eng_rejected)
admin.site.register(Ship_temp)
admin.site.register(Ship_stored)
admin.site.register(Ship_rejected)
admin.site.register(Port)
admin.site.register(Plan_temp)
admin.site.register(Plan_stored)
admin.site.register(Plan_rejected)


