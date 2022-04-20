from rest_framework import routers
from django.urls import path
from . import views

router = routers.DefaultRouter()
router.register(r'Results', views.ResultsViewSet)

urlpatterns = [
    # path('results', views.stock_analyzer, name='results'),
    # path('api', include(router.urls)),
    path('incomplete', views.load_stock, name='')
]

urlpatterns += router.urls
