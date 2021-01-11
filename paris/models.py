from django.db import models


MATCH_STATUS = [('Scheduled', 'Scheduled'),
                ('Live', 'Live'),
                ('Results', 'Results'),
                ('Rescheduled', 'Rescheduled'),
                ('Canceled', 'Canceled'),
                ('Expired', 'Expired'),
                ('Empty', 'Empty')]

BET_STATUS = [('Pending', 'Pending'),
              ('Finished', 'Finished'),
              ('Canceled', 'Canceled')]

WINNER = [('Team 1', 'Team 1'),
          ('Team 2', 'Team 2'),
          ('None', 'None')]

BET_STRATEGY = [('Naive', 'Naive'),
                ('EV', 'EV'),
                ('HLTV', 'HLTV')]

MODELS = [('NN', 'NN'),
          ('XGB', 'XGB'),
          ('HLTV', 'HLTV')]


class Match(models.Model):
    team1 = models.CharField(max_length=200)
    team2 = models.CharField(max_length=200)
    odd1 = models.DecimalField(max_digits=10, decimal_places=2)
    odd2 = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateTimeField('date scheduled')
    logo1 = models.CharField(max_length=200)
    logo2 = models.CharField(max_length=200)
    status = models.CharField(max_length=200, choices=MATCH_STATUS)
    bo = models.CharField(max_length=200)
    match_id = models.IntegerField(primary_key=True)
    winner = models.CharField(max_length=200, choices=WINNER)
    time_to_go = models.CharField(max_length=200)

    def __str__(self):
        return "%s vs %s" % (self.team1, self.team2)


class Prediction(models.Model):
    match = models.OneToOneField(Match,
                                 on_delete=models.CASCADE,
                                 primary_key=True)
    odd1 = models.DecimalField(max_digits=100, decimal_places=2)
    odd2 = models.DecimalField(max_digits=100, decimal_places=2)
    delta_ev_1 = models.DecimalField(max_digits=100,
                                     decimal_places=2,
                                     default=0)
    delta_ev_2 = models.DecimalField(max_digits=100,
                                     decimal_places=2,
                                     default=0)
    model = models.CharField(max_length=200, choices=MODELS)

    def __str__(self):
        return "Prediction of %s" % (self.match)


class Bet(models.Model):
    class Meta:
        constraints = [models.UniqueConstraint(fields=['match', 'strategy'],
                                               name='unique strat')]
    match = models.ForeignKey(Match, on_delete=models.CASCADE)
    winner = models.CharField(max_length=200, choices=WINNER, default='None')
    amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    status = models.CharField(max_length=200,
                              choices=BET_STATUS,
                              default='Pending')
    gain = models.DecimalField(max_digits=10, decimal_places=5, default=0)
    real = models.BooleanField(default=False)
    strategy = models.CharField(max_length=200,
                                choices=BET_STRATEGY,
                                default='EV')
    bet_id = models.AutoField(primary_key=True)

    def __str__(self):
        return "%0.2f €  on %s" % (self.amount,
                                self.match.team1 * (self.winner == 'Team 1')
                                + self.match.team2 * (self.winner == 'Team 2'))
