from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('stocks/', include('alpha_capital.stocks.urls')),
    path('analysis/', include('alpha_capital.analyzer.urls')),
    # path('crawler/', include('alpha_capital.crawler.urls')),
]
