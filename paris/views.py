from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import generic
from django.db.models import Q
from django.utils import timezone
from datetime import datetime, timedelta
import pytz
from .utils import compute_time_to_go
from .forms import BetForm
from .models import Match, Bet, Prediction
import numpy as np


MATCH_NOT_FINISHED = Q(status='Scheduled') | Q(status='Live')
DATA_DIR = 'var/'


class IndexView(generic.TemplateView):
    template_name = 'paris/index.html'
    context_object_name = ''

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        matches = Match.objects.filter(MATCH_NOT_FINISHED).order_by('date')
        compute_time_to_go(matches)
        context['matches'] = matches
        context['bets'] = Bet.objects.order_by('-match__date')
        context['total'] = np.array(list(map(lambda x: x.gain,Bet.objects.filter(status='Finished')))).sum()
        return context


def get_bets_list_by_ev(request):
    ev_trsh = request.GET.get('ev')
    data = {'bets': Bet.objects.filter(ev__gte=ev_trsh)}
    return render(request, '/bet_history', data)


class DayView(generic.ListView):
    template_name = 'paris/day.html'
    context_object_name = 'matches'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        strategy = self.kwargs['strat']
        date_ = self.kwargs['date']
        date_ = datetime.strptime(date_, "%d.%m.%Y")
        tz = timezone.get_current_timezone()
        date_ = tz.localize(date_, is_dst=None)
        print(date_)
        date_next_day = date_ + timedelta(days=1)
        IN_DAY = Q(date__gte=date_, date__lt=date_next_day)
        NOT_BET = Q(status='Rescheduled')|Q(status='Canceled')
        matches = Match.objects.filter(IN_DAY).exclude(NOT_BET).filter(prediction__pk__isnull=False).order_by('-date')
        compute_time_to_go(matches)
 
        if strategy == 'ev':
            bets = Bet.objects.filter(match__in=matches,
                                      strategy='EV').order_by('-match__date')
        elif strategy == 'kelly':
            bets = Bet.objects.filter(match__in=matches,
                                      strategy='Kelly').order_by('-match__date')
        elif strategy == 'naive':
            bets = Bet.objects.filter(match__in=matches,
                                      strategy='Naive').order_by('-match__date')
        print(len(bets),len(matches))
        context['mbs'] = zip(matches, bets)
        return context

    def get_queryset(self):
       
        return None


class BetView(generic.CreateView):
    template_name = 'paris/bet_form.html'
    model = Bet
    form_class = BetForm
    success_url = reverse_lazy('paris:index')


class BetHistory2(generic.ListView):
    template_name = 'paris/bet_history2.html'
    context_object_name = 'bets'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        strategy = self.kwargs['strat']
        context['strategy'] = strategy
        ev = self.kwargs['ev']
        if strategy == 'ev':
            bets = Bet.objects.filter(strategy='EV')

        elif strategy == 'naive':
            bets = Bet.objects.filter(strategy='Naive')

        elif strategy == 'kelly':
            bets = Bet.objects.filter(strategy='Kelly')

        n = 20
        date_0 = timezone.now().replace(second=0, minute=0,
                                        hour=0, microsecond=0)
        dates = [date_0-timezone.timedelta(days=i)
                 for i in range(-1, n)]
        totals = []
        for i in range(0, n+1):
            date_next_day = dates[i] + timedelta(days=1)
            bets_today = (bets.filter(match__date__lt=date_next_day,
                                      match__date__gte=dates[i])
                              .order_by('-match__date'))
            totals.append(np.array(list(map(lambda x: x.gain, bets_today))).sum())
        context['totals'] = dict(zip(dates, totals))
        context['total'] = np.array(totals, dtype=np.float).sum()
        return context

    def get_queryset(self):
        return None


class PredictionView(generic.ListView):
    template_name = 'paris/pred.html'
    context_object_name = 'pred_list'

    def get_queryset(self):
        return (Prediction.objects
                          .order_by('-match__date')
                          .filter(MATCH_NOT_FINISHED))


class HistoryView(generic.ListView):
    template_name = 'paris/history.html'
    context_object_name = 'matches_history'

    def get_queryset(self):
        return Match.objects.exclude(MATCH_NOT_FINISHED).order_by('-date')
