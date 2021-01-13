from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import generic
from django.db.models import Q
from django.utils import timezone
from datetime import datetime, date, timedelta
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
        context['matches'] = Match.objects.filter(MATCH_NOT_FINISHED).order_by('date')
        context['bets'] = Bet.objects.order_by('-match__date')
        context['total'] = np.array(list(map(lambda x: x.gain,Bet.objects.filter(status='Finished')))).sum()
        return context


def get_bets_list_by_ev(request):
    ev_trsh = request.GET.get('ev')
    data = {'bets': Bet.objects.filter(ev__gte=ev_trsh)}
    return render(request, '/bet_history', data)


class UpcomingView(generic.ListView):
    template_name = 'paris/upcoming.html'
    context_object_name = 'matches_upcoming'

    def get_queryset(self):
        matches = Match.objects.filter(MATCH_NOT_FINISHED).order_by('date')
        compute_time_to_go(matches)
        return matches


class DayView(generic.ListView):
    template_name = 'paris/day.html'
    context_object_name = 'matches'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        strategy = self.kwargs['strat']
        date = self.kwargs['date']
        date = datetime.strptime(date, "%d.%m.%Y")
        date = pytz.timezone("Europe/Paris").localize(date, is_dst=None)
        date_next_day = date + timedelta(days=1)
        IN_DAY = Q(date__gte=date, date__lt=date_next_day)
        matches = Match.objects.filter(IN_DAY).order_by('-date')
        if strategy == 'ev':
            bets = Bet.objects.filter(match__in=matches,
                                      strategy='EV').order_by('-match__date')
        elif strategy == 'kelly':
            bets = Bet.objects.filter(match__in=matches,
                                      strategy='Kelly').order_by('-match__date')
        elif strategy == 'naive':
            bets = Bet.objects.filter(match__in=matches,
                                      strategy='Naive').order_by('-match__date')
        context['mbs'] = zip(matches, bets)
        return context

    def get_queryset(self):
        date = self.kwargs['date']
        date = datetime.strptime(date, "%d.%m.%Y")
        date = pytz.timezone("Europe/Paris").localize(date, is_dst=None)
        date_next_day = date + timedelta(days=1)
        IN_DAY = Q(date__gte=date, date__lt=date_next_day)
        matches = Match.objects.filter(IN_DAY).order_by('-date')
        compute_time_to_go(matches)
        return matches


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
                 for i in range(n)]
        totals = []
        for i in range(n-1):     
            date_next_day = dates[i] + timedelta(days=1)
            bets_today = (bets.filter(Q(match__prediction__delta_ev_1__gte=ev) |
                                      Q(match__prediction__delta_ev_2__gte=ev))
                              .filter(match__date__lt=date_next_day,
                                      match__date__gte=dates[i])
                              .order_by('-match__date'))
            totals.append(np.array(list(map(lambda x: x.gain, bets_today))).sum())
        context['totals'] = dict(zip(dates, totals))
        context['total'] = np.array(totals, dtype=np.float).sum()
        return context

    def get_queryset(self):
        strategy = self.kwargs['strat']
        if strategy == 'ev':
            ev = self.kwargs['ev']

            bets = (Bet.objects.filter(strategy='EV')
                    .filter(Q(match__prediction__delta_ev_1__gte=ev) |
                            Q(match__prediction__delta_ev_2__gte=ev))
                    .order_by('-match__date'))
            return bets

        elif strategy == 'naive':
            bets = Bet.objects.filter(strategy='Naive').order_by('-match__date')
            return bets

        elif strategy == 'kelly':
            bets = Bet.objects.filter(strategy='Kelly').order_by('-match__date')
            return bets


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
