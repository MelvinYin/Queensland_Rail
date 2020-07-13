from django import forms
from QRvisualisation.models import FileInput,MaintenanceZonalFileInput
#from crispy_forms.helper import FormHelper

class DocumentForm(forms.ModelForm):
    class Meta:
        model = FileInput
        fields = ('output_name', 'skiprows','TRC_file')


class MaintenanceZonalForm(forms.ModelForm):
    class Meta:
        model = MaintenanceZonalFileInput
        fields = ('num_quarters','line')
