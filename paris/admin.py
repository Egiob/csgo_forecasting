from django.contrib import admin

from .models import Match,Bet,Prediction

admin.site.register(Match)
admin.site.register(Bet)
admin.site.register(Prediction)