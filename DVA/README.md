# Queensland Rail Maintenance Prediction and Visualisation Project
[http://ec2-3-18-150-48.us-east-2.compute.amazonaws.com:8000/]

## DESCRIPTION

Mostly Python3-based app, developed in close collaboration with and targeted for Queensland Rail. Takes in as input raw Track-Recording-Car (TRC) data, and attendant data, including terrain condition and quality, GPS positions of culverts, and location and timestamps of work orders. Performs alignment of datasets with disparate position data types, such as metrange or position along track for some, and GPS coordinates for others, to a singular position axis column. Also performs alignment for datasets of the same type but collected at different time periods, leading to coordinate drift which has to be compensated for.

Analysis performed include the engagement of different ML models (regression, random trees and forests, neural networks, etc) to provide prediction of track deterioration and consequently identification of locations where track maintenance will be necessary. The need for these are identified during direct communication with QR engineers and executives, preliminary ascertainment of viability is performed usually by manual inspection of preliminary results, and final performance are reported using standard ML metrics, such as prediction accuracy and residual loss. 

Visualisations are built using interactive Bokeh plots and static images, both embedded in Django. These are provided to our client primarily through an AWS webpage, and secondarily through a cross-platform docker container and Linux-specific build instructions. Analysis reports are included in the webpage for consultation. Optimisation for display load times is necessary because of the large input datasets, and are performed while minimising the deterioration in quality of the plots and utility to the users. These can be turned off trivially in the code if necessary, such as in local deployments where network RTT and bandwidth considerations are irrelevant. 

Live update functionalities include user-uploadable TRC datasets, where they are first aligned to existing datasets. Prediction for maintenance hotspots is then performed using a trained random forest model, and results are delivered on a plot showing relative positions of the hotspots on the track, on a geographic coordinate system. These also include prediction on a quarterly basis, of the upcoming maintenance load along different zones on the track, for two tracks (C138 and C195). 

![Home interface](/CODE/demo/main.gif)

## Key Features
* Interactive visualisations of:
  1. Track Geometry, as calculated from combined standard deviation metric of measures derived from Track Recording Car (TRC) data

  ![TRC data](/CODE/demo/TRC_vis.gif)

  2. Geospatial view of Ground penetrating radar (GPR) data

  ![GPR data](/CODE/demo/GPR_vis.gif)

* User-input form for predicting upcoming maintenance needs from the TRC+GPR input. Behind the hood lies a random forest that takes the test data (try it out, using our test data in /test_data)

  ![ML](/CODE/demo/predict_maintenance.gif)

  * User-input form for predicting upcoming maintenance needs from previous maintenance work. Behind the hood lies a random forest.
  
    ![ML](/CODE/demo/zonal_vis.gif)

* Explore our ML reports on the data

  ![reports](/CODE/demo/Reports.gif)

* AWS-EC2 Deployment at ec2-3-18-150-48.us-east-2.compute.amazonaws.com:8000

## INSTALLATION
### Option 1: Manually launching Django app
* This app was built in python 3.6.+. Library requirements in requirements.txt (use pip install requirements.txt)

* Unzip all files in /data ; /models ; src/QRDVA/QRvisualisation/templates/QRvisualisation

* Download the random_forest serialised file here, unzip using 7zip, and place it inside /models.
https://1drv.ms/u/s!AliOttxeJ8OYgYFg_OsX8r0oqBdl9Q?e=cQaozi

1. To initiate app, navigate to src/QRDVA in console, then: $python manage.py makemigrations

2. $ python manage.py migrate

3. $python manage.py runserver

4. In your browser, navigate to localhost:8000/

### Option 2:  Docker Image Download

1. Download compressed version

* Link: https://1drv.ms/u/s!AliOttxeJ8OYgYFhyzaSyQxs39NSow?e=Q2M21H

2. Pull from docker-hub

* docker pull melvinyin/gatech_dva:deploy

Platform-specific instructions for launching a docker container can be found online. To run, cd into project_dir inside container and run python src/QRDVA/manage.py runserver 0.0.0.0:8000. Open 172.17.0.2:8000 (or localhost:8000) in host machine browser to launch app. 

### Option 3: Docker Container Setup

(Working as of 26/11/2019)

Instructions for building a Docker container using the repository on a fresh Ubuntu 18.04 platform:

Abbreviation: bash on host_machine = host; terminal in container = con; container_id = con_id

For convenience, we rely on anaconda-p3 for python setup.

* Download anaconda-p3, either git pull or download repository .zip file

* (host) sudo apt install docker

* (host) docker pull ubuntu

* (host) docker run -it -p 8000:8000 ubuntu

* (con) apt update

* (con) apt install unzip

* (con) apt install p7zip-full p7zip-rar

* In host, download serialised compressed random_forest_model (rf_cv_wo.7z) from gdrive link above

* (host) sudo docker container ls => get container_id

* (host) sudo docker cp Anaconda.sh con_id:/home

* (host) sudo docker cp dva_project.zip con_id:/home

* (host) sudo docker cp rf_cv_wo.7z con_id:/home

* Install anaconda, (con) sh Anaconda.sh => yes

* Init conda, (con) cd /root/anaconda3/bin => ./conda init => exec bash (to restart shell)

* In conda_env, (con) conda install django geopandas descartes

* (con) unzip dva_project.zip

* (con) 7z a rf_cv_wo.7z

* (con) mv rf_cv_wo.pickle ./dva_project/models/

* (con) cd dva_project

* (con) python src/QRDVA/manage.py makemigrations

* (con) python src/QRDVA/manage.py migrate

* (con) python src/QRDVA/manage.py runserver 0.0.0.0:8000

* In host machine, navigate to browser, type 172.17.0.2:8000 (or localhost:8000)

Common errors and potential fixes:

Error: Host machine browser 172.17.0.2:8000 doesn't load

Cause: Either docker container doesn't expose port to host (should be default), or port number is wrong.

Fix: For former, Google, fix depends on host machine. If it's a wrong docker setup then reinstalling docker may help.

For latter, from host bash, while docker container is running, sudo docker inspect con_id => see if ip_address is 172.17.0.2, it's somewhere at the bottom of the returned message. If it isn't, copy this ip_address.

* (con) In dva_project, nano src/QRDVA/QRDVA/settings.py

* (con) Search for ALLOWED_HOSTS, add ip_address in list as string

* (con) Redo python src/QRDVA/manage.py runserver 0.0.0.0:8000

* In host, navigate to browser, type ip_address:8000

Error: apt update fails in docker_container

Cause: Docker container doesn't have internet access, either host machine is disconnected or docker configuration is wrong, since by default it should be turned on.

Fix: Google, fix depends on host machine.

## EXECUTION

To access: See AWS-EC2 website link, INSTALLATION/Docker Image Download 

### Web application
Navigate around in browsesr tab, links to main visualsiations are contained in the top banner.

#### Track geometry and GPR visualisation
1. click on 'visualise track data' on homepage
2. Interactive plots availabe for GPR and track geometry according to two links in page

#### Predict maintenance
1. click on 'Predict track maintenance" on homepage
2. Enter '0', '761001', and upload either test_file1.csv or test_file2.csv (contained in /CODE/test_data/)
3. click submit

#### Predict Zone-wise total maintenance count
1. click on 'predict zone-wise total maintenance count' on homepage
2. enter desired number of quarters ahead to predict (1 to 4) and either 'C138' or 'C195' (corresponding track to perfom prediciton)
3. Click submit


### Utilities
In creating this app using data provided by QR, we created a few tools to clean and join the data.
1. Alignment tool:
* To align TRC runs across time, the python script looks at differences between the Guage and Super values from each run of TRC and minimises the standard deviation between them
* Input: folder with all TRC files (.csv)
* Output: dictionary of pandas dataframes with all TRC aligned in format {'track_code': pandas.dataframe}
* Usage:
from join_trc import align_trc

df = align_trc(path_to_TRC_files, skip=number_of_rows_to_skip).main_align()

2. Joining tool:
* To join TRC data with culverts data, speed data, GPS coordinates data (through interpolation) and GPR data
* input: TRC dataframe (output of align_trc)
* output: dictionary of joined dataframes

usage:
from join_trc import join_trc

joined = join_trc(df, wo_df,  gps_df, culvert_df, speed_df, gpr_df, tcode).main_join()
