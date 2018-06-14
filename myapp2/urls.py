from django.conf.urls import url
from .views import *
urlpatterns=[
    url(r'^inter/$',inter_views)
]