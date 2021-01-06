# CSGO Odds forecasting

The goal of this project is to build an application that provides an interface for forecasting the outcome of Counter Strige: Global Offensive competitive games.
It includes :
- a theoretical approach of gambling theory in the case of betting on competitve games.
- a way to aggregate different sources available online to build features to predict the outcomes of the matches.
- a model and a decision rule that is supposed to beat a particular bookmaker, only using the latter features.
- some tools and simulations of what can be done in the field of Counter Strike odds prediction.

You can find a whole detailled explanation of the content of the project [here](report.pdf).

If you have any questions feel free to contact me !

### If you want to test it yourself
 - Make sure you use shell_plus when you run the notebooks (see : [how-to-use-django-in-jupyter-notebook](https://medium.com/ayuth/how-to-use-django-in-jupyter-notebook-561ea2401852))
 ```python 
 python manage.py shell_plus --notebook
 ```
 - Make sure you git is configured to keep symbolic links
 ```
 git config --global core.symlinks true
 ```
 
 - Make sure you have a good version of chromedriver in your PATH (see : [chromedriver](https://chromedriver.chromium.org/))
 - When you run migrations for the first time use:
 ```python
 python manage.py migrate --run-syncdb
 ```
