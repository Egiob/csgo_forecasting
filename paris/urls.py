from django.urls import path

from . import views


app_name = 'paris'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('history', views.HistoryView.as_view(), name='history'),
    path('upcoming', views.UpcomingView.as_view(), name='upcoming'),
    path('bet',views.BetView.as_view(),name='bet'),
    path('bet_history2/<str:ev>/<str:strat>',
         views.BetHistory2.as_view(),
         name='bet_history2'),
    path('pred', views.PredictionView.as_view(), name='pred'),
]
