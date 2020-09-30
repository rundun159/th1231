from django.forms import ModelForm
from .models import *
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import get_user_model

class MateForm(ModelForm):
    class Meta:
        model = Mate_temp
        fields = ['mate_name', 'mate_comp', 'gender', 'age']
        labels = {
            'mate_name': _('name of new mate'),
            'mate_comp': _('Company of new mate'),
            'gender': _('Gender of new mate'),
            'age': _('Age of new mate')
        }
class EngForm(ModelForm):
    class Meta:
        model = Eng_temp
        fields = ['eng_name', 'eng_comp', 'gender', 'age']
        labels = {
            'eng_name': _('name of new eng'),
            'eng_comp': _('Company of new eng'),
            'gender': _('Gender of new eng'),
            'age': _('Age of new eng')
        }
class ShipForm(ModelForm):
    class Meta:
        model = Ship_temp
        fields = ['ship_name', 'ship_comp', 'age']
        labels = {
            'ship_name': _('name of new ship'),
            'ship_comp': _('Company of new ship'),
            'age': _('Age of new eng')
        }
class PlanForm(ModelForm):
    class Meta:
        model = Plan_temp
        fields = ['plan_port1', 'plan_port2', 'departure', 'arrival', 'plan_ship', 'plan_comp']
        labels = {
            'plan_port1': _('Starting Port'),
            'plan_port2': _('Ending Port'),
            'departure': _('The day plan starts'),
            'arrival': _('The day plan ends'),
            'plan_ship': _('Ship'),
            'plan_comp': _('Company'),
        }
class SigUpForm(ModelForm):
    class Meta:
        model = Comp
        fields = ['comp_name', 'comp_id', 'comp_passwd']
        labels = {
            'comp_name': _('name of new Company'),
            'comp_id': _('ID of new Company'),
            'comp_passwd': _('Password of new Company'),
        }
class Comp_LogInForm(ModelForm):
    class Meta:
        model = Comp
        fields = ['comp_id', 'comp_passwd']
        labels = {
            'comp_id': _('ID of Company'),
            'comp_passwd': _('Password of Company'),
        }

class AddNewPortForm(ModelForm):
    class Meta:
        model = Port
        fields = ['port_name']
        labels = {
            'port_name': _('New port'),
        }

class ModifyPlanForm(ModelForm):
    class Meta:
        model = Modify_Plan
        fields = ['departure', 'arrival']
        labels = {
            'departure': _('Modify Departure'),
            'arrival': _('Modify Arrival')
        }
