from django import forms

class PasswordForm(forms.Form):
    password = forms.CharField(label='Password', max_length=100)