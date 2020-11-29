from django import forms
from .models import Bet,Match

from django.db.models import Q

class BetForm(forms.ModelForm):


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fields['match'].queryset = Match.objects.filter(Q(status='Scheduled')|Q(status='Live')).order_by('date')

		'''if 'match' in self.data:
			try:

				match_id = int(self.data.get('match'))
				team1 = Match.objects.filter(match_id=match_id)[0].team1.strip()
				team2 = Match.objects.filter(match_id=match_id)[0].team2.strip()
				self.fields['winner'].choices = [(team1,team1),(team2,team2)]
				print([(team1,team1),(team2,team2)])
				

			except (ValueError, TypeError):
				pass  # invalid input from the client; ignore and fallback to empty City queryset
				'''


	class Meta:
		model = Bet
		fields = ['amount','match','winner']
		

