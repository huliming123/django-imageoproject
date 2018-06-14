from django.conf.urls import url
from .views import *
urlpatterns=[
    url(r'^index/$',index_views),
    url(r'^login/$',login_views),
    url(r'^$',final_views)
]