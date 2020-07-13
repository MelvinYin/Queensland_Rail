from django.shortcuts import render_to_response
from django.shortcuts import render, redirect
# for forms
from django.views.generic import FormView
from .forms import DocumentForm,MaintenanceZonalForm
#from .models import qrOutput
# for bokeh

from .bokeh_ui.figures.user_heatmap import *
from .bokeh_ui.main_ui import *

## for reading csv
## joins
from .utils.to_gps_coords import *
from .utils.preprocess import *

## for transforming to geodata
import geopandas as gpd
from shapely.geometry import Point

## for loading model
import pickle

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

import os

## initiate vars
distance = 0
num_wo = 0
num_quarters=0

# Create your views here.
def index(request):
    return render(request, 'QRvisualisation/index.html', {})


def output_visualisation(request):
    #p = show_plot()
    #script, div = components(p)
    #return render_to_response( 'QRvisualisation/QRplot.html', {'resources': CDN.render(), 'script' : script, 'div': div})
    return render_to_response('QRvisualisation/QRplot.html', {})

def output_visualisation_2(request):
    global distance, num_wo
    #p = show_plot()
    #script, div = components(p)
    return render_to_response( 'QRvisualisation/output_maintenance.html', {'dist': distance, 'num_wo':num_wo})


def output_visualisation_maintenance_zonal(request):
    #p = show_plot()
    #script, div = components(p)
    return render_to_response( 'QRvisualisation/output_maintenance_zonal.html', {'num_quarters':num_quarters})

def descriptive_stats_marcus(request):
    return render_to_response('QRvisualisation/C139analysis.html', {})

def descriptive_stats_jj(request):
    global distance, num_wo
    return render_to_response('QRvisualisation/woPrediction.html', {})

def descriptive_stats_rahul(request):
    return render_to_response('QRvisualisation/C138_C195analysis.html', {})

def output_trc_vis(request):
    return render_to_response('QRvisualisation/QRplot_trc_wrapper.html')


def model_output(trc, track_code):
        gps_data = pd.read_csv(os.path.join(paths.DATA_DERIVED, 'C138_C195_coords.csv'))
        ## extract features
        to_predict = preprocess_x(trc)
        ## load model from file
        with open(os.path.join(paths.MODELS, 'rf_cv_wo.pickle'), 'rb') as f:
            loaded_model = pickle.load(f)
        result = loaded_model.predict(to_predict)
        ## join back on data
        trc['predicted_output'] = result
        ## generate gpsdata
         ## state geometry

        ans_df = trc[['METRAGE', 'predicted_output', 'Dec.Long', 'Dec.Lat']].rename(columns={'Dec.Long':'longitude', 'Dec.Lat':'latitude'})
        #ans_df = ans_df.fillna(0)
        geometry = [Point(xy) for xy in zip(ans_df['longitude'], ans_df['latitude'])]
        geo_df = gpd.GeoDataFrame(ans_df, crs = {'init':'epsg:4326'}, geometry = geometry) # create geopandas df

        dist = len(trc)
        num_wo = len(np.where(result == 1)[0].tolist())

        return geo_df, dist, num_wo


def model_output_maintenance_zonal(num_quarters,line):
    print("here for quarters = {} line = {}".format(num_quarters,line))
    with open(os.path.join(paths.MODELS, 'rf_maintenance_zonal_' + line + '.pickle'), 'rb') as f:
        loaded_model = pickle.load(f)

    y = pd.DataFrame(columns=[i for i in range(23, 23 + int(num_quarters))], index=[i for i in range(1, 101)])
    x = np.ndarray((0, 0))
    for zone in range(1, 101):
        for quarter in range(23, 23 + int(num_quarters)):
            if x.shape[0] == 0:
                x = np.array([quarter % 4, zone])
            else:
                x = np.vstack((x, [quarter % 4, zone]))
    y_pred = loaded_model.predict(x)
    index = 0
    for zone in range(1, 101):
        for quarter in range(23, 23 + int(num_quarters)):
            y.loc[zone, quarter] = int(y_pred[index])
            index += 1
    return y


def create_user_heatmap(df):
    fig, ax = plt.subplots(figsize = (20,5))
    ax = sns.heatmap(df[['predicted_output']].as_matrix().T, cmap = 'Oranges')
    curr_path = os.getcwd()
    #output_path = os.path.join(curr_path, 'QRvisualisation', 'templates', 'QRvisualisation','images')
    output_path = os.path.join(curr_path, 'media')
    print(output_path)
    plt.savefig(os.path.join(output_path, 'pred_heatmap.png'))

def create_heatmap_maintenance_zonal(y):
    plt.figure()
    plt.pcolor(y, cmap = 'Blues')
    plt.xlabel("Quarter")
    plt.ylabel("Zone")
    # plt.draw()
    curr_path = os.getcwd()
    #output_path = os.path.join(curr_path, 'QRvisualisation', 'templates', 'QRvisualisation','images')
    output_path = os.path.join(curr_path, 'media')
    print("here. shape of y = {}".format(y.shape))
    print(output_path)
    #if os.path.exists(curr_path+ '/src/QRDVA/media/pred_heatmap_maintenance_zonal.png'):
    #    os.remove(curr_path+ '/src/QRDVA/media/pred_heatmap_maintenance_zonal.png')
    # plt.savefig(os.path.join(output_path, 'pred_heatmap_maintenance_zonal.png'))
    plt.savefig(curr_path+ '/src/QRDVA/media/pred_heatmap_maintenance_zonal.png')
    plt.savefig(os.path.join(output_path, 'pred_heatmap_maintenance_zonal.png'))


class FormView(FormView):
    template_name = 'QRvisualisation/form.html'
    form_class = DocumentForm
    success_url = 'QRvisualisation/output'
    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        TRC = self.request.FILES['TRC_file']
        skiprows = self.request.POST['skiprows']
        track_code = self.request.POST['output_name']
        global TRC_file
        ## load the uploaded data
        TRC_file = pd.read_csv(TRC, skiprows = int(skiprows))
        ## get model output
        global output, distance, num_wo
        output, distance, num_wo = model_output(TRC_file, track_code)

        create_user_heatmap(output)

        #print(output.shape)
        #print(output.head())
        if TRC.name.endswith('csv') : #and act_file.name.endswith('html')
            form.save()
            trc = os.path.join(os.getcwd(), 'documents', TRC.name.replace(' ', '_'))
            try:
                make_user_wo_plot(output,
                              "./src/QRDVA/QRvisualisation/templates/QRvisualisation"
                              "/output.html")
            except FileNotFoundError:
                make_user_wo_plot(output, "QRvisualisation/templates/QRvisualisation/output.html")

        else:
            return redirect('QRvisualisation:formview')

        return redirect('QRvisualisation:outputview_2')



class FormViewMaintenanceZonal(FormView):
    template_name = 'QRvisualisation/maintenance_zonal_form.html'
    form_class = MaintenanceZonalForm
    success_url = 'QRvisualisation/output_maintenance_zonal'
    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.

        global output, num_quarters

        num_quarters = self.request.POST['num_quarters']
        line = self.request.POST['line']

        if not num_quarters.isdigit() or int(num_quarters)<=0 or int(num_quarters)>4:
            print("in redirect to same page")
            return redirect('QRvisualisation:formview_maintenance_zonal')

        output = model_output_maintenance_zonal(num_quarters,line)

        create_heatmap_maintenance_zonal(output)

        #print(output.shape)
        #print(output.head())
        form.save()


        return redirect('QRvisualisation:outputview_maintenance_zonal')
