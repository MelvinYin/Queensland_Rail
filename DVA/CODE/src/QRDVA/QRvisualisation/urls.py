
from django.urls import path, reverse
from django.conf.urls  import url
from .views import *
#compareOutputView
app_name = 'QRvisualisation'

urlpatterns = [
    path('', index, name = 'index'),
    path('output/', output_visualisation, name = 'outputview'),
    path('form/', FormView.as_view(), name = 'formview'),
    path('output_maintenance/', output_visualisation_2 , name = 'outputview_2'),
    path('C139_analysis/', descriptive_stats_marcus, name = 'descriptive_stats_marcus'),
    path('WO_prediction/', descriptive_stats_jj, name = 'descriptive_stats_jj'),
    path('WO_by_quarter/', descriptive_stats_rahul, name = 'descriptive_stats_rahul'),
    path('output_trc/', output_trc_vis, name = 'output_trc_vis'),
    path('form_maintenance_zonal/', FormViewMaintenanceZonal.as_view(), name = 'formview_maintenance_zonal'),
    path('output_maintenance_zonal/', output_visualisation_maintenance_zonal, name = 'outputview_maintenance_zonal'),

]

## path('desired path', view class, name of view)
