import requests

from django.shortcuts import render

from .forms import PasswordForm

def get_password(request):
    if request.method == 'POST':
        form = PasswordForm(request.POST)
        if form.is_valid():
            password_weakness = requests.post(
                'http://127.0.0.1:5000/predict-password-complexity-v1', 
                data={'password': form.cleaned_data['password']},
            ).json()['prediction']
            
            variables = {'form': form, 'password_weakness': password_weakness}
            return render(request, 'password_form.html', variables)
    else:
        form = PasswordForm()
    return render(request, 'password_form.html', {'form': form})
