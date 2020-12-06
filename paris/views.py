from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import generic
from django.db.models import Q
from django.utils import timezone

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
        if strategy == 'ev':
            ev = self.kwargs['ev']
            n = 200
            dates = [timezone.now()-timezone.timedelta(days=i)
                     for i in range(n)]
            totals = []
            for i in range(n-1):
                date_sup = timezone.now()-timezone.timedelta(days=i)
                date_inf = timezone.now()-timezone.timedelta(days=i+1)
                bets = Bet.objects.filter(strategy='EV')
                bets = (bets.filter(Q(match__prediction__delta_ev_1__gte=ev) |
                                    Q(match__prediction__delta_ev_2__gte=ev))
                            .filter(match__date__lte=date_sup,
                                    match__date__gte=date_inf)
                            .order_by('-match__date'))
                totals.append(np.array(list(map(lambda x: x.gain, bets))).sum())
            context['totals'] = dict(zip(dates, totals))

        else:
            n = 200
            dates = [timezone.now()-timezone.timedelta(days=i)
                     for i in range(n)]
            totals = []
            for i in range(n-1):
                date_sup = timezone.now()-timezone.timedelta(days=i)
                date_inf = timezone.now()-timezone.timedelta(days=i+1)
                bets = Bet.objects.filter(match__date__lte=date_sup,
                                          match__date__gte=date_inf,
                                          strategy='Naive').order_by('-match__date')
                totals.append(np.array(list(map(lambda x: x.gain,
                                                bets))).sum())
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
        else:
            bets = Bet.objects.filter(strategy='Naive').order_by('-match__date')
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
