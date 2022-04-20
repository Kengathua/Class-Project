#========================================= DRF (REST urls)=========================================
from django.urls import include, path
from rest_framework import routers
from alpha_capital.stocks.views import (
    TickerViewSet, StockViewSet, nse_crawler)

from django_filters.rest_framework import DjangoFilterBackend

router = routers.DefaultRouter()
router.register(r'Tickers', TickerViewSet)
router.register(r'Stocks', StockViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    #path('api-auth/', include('rest_framework.urls', namespace='api_rest_framework'))
]

urlpatterns +=[
    # path('form/', views.FormView, name='form'),
    path('live_crawler/', nse_crawler, name='live_crawler')
]

