import os
from utils import paths
import pandas as pd
import numpy as np
from ast import literal_eval
from to_gps_coords import *

'''
Takes TRC, GPS, culvert, speed data, aligns TRC data then joins
based on track code
'''
class align_trc():
    def __init__(self, path_folder, odd_filenames = False, skip = 0):
        self.skip = skip ## numrows to skip
        self.odd_filenames = odd_filenames # rename odd filenames and add to a sub-folder 'combined'
        self.path_folder = path_folder # add path to file list


    def get_df(self):
        return self.combined_df

    def filelist_df(self, folder):
        print('Preparing list of files')
        files = os.listdir(folder)
        filedf = []
        for f in files[1:]:
            ans = dict()
            br = f.split('-')[:-1]
            ans['Track_ID']= br[0].replace(' ', '')
            ans['Description'] = '-'.join(br[1:-2])
            ans['Date'] = br[-1].replace(' ', '')[:8]
            ans['Recording_car']= br[-1].replace(' ','')[8:]
            ans['filename'] = f
            filedf.append(ans)
        f_df = pd.DataFrame(filedf)

        if self.odd_filenames == True:
            c_files = []
            ## add odd files into the combined folder, with format trackcode_date.csv
            cfiles = os.listdir(os.path.join(self.path_folder, 'combined'))
            for f in cfiles:
                Track_ID = f.split('_')[0]
                Date = f.split('_')[1].split('.')[0]
                a = {'Track_ID': Track_ID, 'Description' : '', 'Date': Date, 'Recording_car':'', 'filename': f}

            c_files.append(a)
            file_df = pd.concat([f_df, pd.DataFrame(c_files)])
        print('prepared list of files')
        print(f_df)
        return f_df

    def read_files(self, file_df):
        print('reading files')
        final_df = dict()
        unique_track_id = file_df['Track_ID'].unique()
        print(unique_track_id)
        for ind in unique_track_id:
            total = []
            # iterate through files one track id at a time
            for i, file in file_df[file_df['Track_ID']== ind].iterrows():
                df = pd.read_csv(os.path.join(self.path_folder, file['filename']), skiprows = self.skip)
                df['Date'] = file['Date'] #assign date
                total.append(df)
            total_df = pd.concat(total)
            ## now reshape the table
            long_df = pd.DataFrame(total_df.set_index(['Date','METRAGE']).stack()).reset_index()
            long_df.columns = ['Date', 'METRAGE', 'Measure','Value']
            temp = long_df.set_index(['Measure','METRAGE','Date'])
            df_wide = temp.loc[~temp.index.duplicated(keep='first')].unstack('Date')
            df_wide.columns = df_wide.columns.droplevel(0)
            df_wide = df_wide.reset_index()
            final_df[ind] = df_wide.set_index('METRAGE')
        print('combined files into a master df')
        self.combined_df = final_df

    # def check_anomaly(self, df):
    #
    #     dates = df.columns[1:-1]
    #     metrange_range = {date: {'minimum': 0, 'maximum': 0, 'nrow': 0} for date in dates}
    #     mins = []
    #     maxs = []
    #     for date in dates:
    #         temp = df[['METRAGE', date]].dropna()
    #         temp[date]['minimum'] = min(temp['METRAGE'])
    #         temp[date]['maximum'] = max(temp['METRAGE'])
    #         temp[date]['nrow'] = len(temp)
    #         mins.append(min(temp['METRAGE']))
    #         maxs.append(max(temp['METRAGE']))
    #
    #     ## check if ranges overlap
    #     assert len(mins) == len(maxs)
    #     for i in range(len(mins)):
    #         for j in range(len(max)):
    #             if (maxs[j] < min[i]) or (mins[j] > maxs[i]):
    #                 print()
    #
    #


    ## functions to align trc_df
    def align_TRC_all(self, df, date1, date2):
        print('Comparing {0} with {1}'.format(date1, date2))
        measures = ['GAUGE', 'SUPER']
        meas_min = dict()
        #
        # l1 = len(df[['METRAGE', date1]].dropna()['METRAGE'])
        # l2 = len(df[['METRAGE', date2]].dropna()['METRAGE'])
        # if l2 < l1: #shorter
        #     r = round(0.1 * l2)
        # else:
        #     r = round(0.1 * l1)
        # print(r)
        # assert r > 50

        for m in measures:
            #print('Looking at measure: {0}'.format(m))
            counts = []
            temp_df = df[df['Measure']== m][[date1, date2]]
            #print(temp_df.shape)
            for i in range(100):
                val1 = temp_df.iloc[500:-500, :][date1].values
                val2 = temp_df.iloc[450+i:-550+i, :][date2].values
                diff1 = val1 - val2
                ans = np.std(diff1[~np.isnan(diff1)])
                counts.append(ans)
            min_std = np.argmin(np.array(counts))
            meas_min[m]=min_std-50
        g = list(meas_min.values())

        rec = round(sum(g)/len(g)) # return the mode of alignment
        #print('recommended:shift {0} by {1} compared to {2} '.format(date2, rec, date1))
        return rec

    def shift_TRC(self, df, date, shift_value):
        return df.groupby('Measure')[date].shift(shift_value)

    def fill_method(self, col, nrow):
        try:
            ans = (col.fillna(method = 'ffill', limit = nrow))
        except: ## too many missing values
            ans = np.nan
        return ans


    def align_TRC_main(self, df):
    ## takes a list of dataframes and aligns them
        data_raw = df
        data =  df.groupby('Measure').transform(lambda x: x.fillna(x.mean())) ## fill NA values with mean val

        data['Measure'] = data_raw['Measure']
        dates = [date for date in data.columns.tolist() if date !='Measure' and date != 'METRAGE'] # get a list of dates
        ans = {'METRAGE' : data['METRAGE'].values,
                   'Measure': data['Measure'].values}
        r = len(dates)
        offset = [0]

        for i in range(1,r):
            rec = self.align_TRC_all(data, dates[i-1], dates[i])
            offset.append(rec)
        to_offset = [sum(offset[:ind+1]) for ind, val in enumerate(offset)]
        print(offset)
        print(to_offset)
        # offset all
        ## first define actual dataset to shift

        for ind, offset_by in enumerate(to_offset):
            print(dates[ind], offset_by)
            x = self.shift_TRC(data_raw, dates[ind], offset_by)
            ans[dates[ind]] = x
        ans_shifted = pd.DataFrame(ans).groupby('Measure').transform(lambda x: x.fillna(self.fill_method(x, 6)))
        ans_shifted['Measure'] = data_raw['Measure']
        return ans_shifted

    def main_align(self):
        self.trc_df = self.read_files(self.filelist_df(self.path_folder))
        total_df = dict()
        for key, table in self.combined_df.items():
            print('='*100)
            print('aligning {}'.format(key))
            total_df[key] = self.align_TRC_main(table.reset_index())
        return total_df



class join_trc():
    def __init__(self, trc_df, wo_df,  gps_df, culvert_df, speed_df, gpr_df, track_code, path_folder = paths.DATA_DERIVED):
        self.track_code = track_code
        self.TRC = self.pivot_to_measure(trc_df)
        self.path_folder = path_folder # add path to file list
        self.wo_df = pd.read_csv(os.path.join(self.path_folder, wo_df))
        #print(self.wo_df.shape)
        self.gps_df = pd.read_csv(os.path.join(self.path_folder, gps_df))
        self.culvert_df = pd.read_csv(os.path.join(self.path_folder,culvert_df))
        self.speed_df =pd.read_csv(os.path.join(self.path_folder,speed_df))
        self.gpr_df = pd.read_csv(os.path.join(self.path_folder, gpr_df))
         # only first 3 digits

    def pivot_to_measure(self, df):
        cols = ['METRAGE','Measure', 'Date', 'Value']
        stacked = pd.DataFrame(df.set_index(['METRAGE', 'Measure']).stack()).reset_index()
        stacked.columns = cols
        pivoted = pd.pivot_table(stacked, index = ['METRAGE' , 'Date'], values='Value', columns = "Measure").reset_index()
        pivoted['Track_code'] = self.track_code
        return pivoted

    def extract_track_code_wo(self, x):
        try:
            ans = literal_eval(x)[0][:3]
        except:
            ans = ''
        return ans

    def get_maximum_date(self, x):
        try:
            ans = max(x)
        except:
            ans = ''
        return ans

    def match_work_orders(self):
        self.TRC['Date'] = pd.to_datetime(self.TRC['Date'])
        self.TRC['Work_orders'] = np.empty((len(self.TRC), 0)).tolist()
        self.TRC['Work_order_type'] = np.empty((len(self.TRC), 0)).tolist()

        datelist = sorted(self.TRC['Date'].unique(), reverse= True)

        w_o = self.wo_df.loc[self.wo_df['Track_code'] == self.track_code[:3], :]
        print('found {} work orders to join'.format(len(w_o)))
        w_o.loc[:, 'check_Date'] = pd.to_datetime(w_o['Bas. start date'])
        w_o['Date'] = np.empty((len(w_o), 0)).tolist()
        print(w_o.head())

        for date in datelist:
            #print(len(w_o.loc[w_o['check_Date']> date, 'Date']))
            w_o.loc[w_o['check_Date']> date, 'Date'].apply(lambda x: x.append(date))

        print(len(w_o[w_o['Date'].apply(lambda x: len(x))== 0]))
        w_o.loc[:, 'Date'] = w_o['Date'].apply(lambda x: self.get_maximum_date(x))
        w_o.loc[:, 'Start Point'] = w_o['Start Point'].apply(lambda x: float(str(x).replace(',', '')))
        w_o.loc[:, 'End Point'] = w_o['End Point'].apply(lambda x: float(str(x).replace(',', '')))

        for i, workorder in w_o.iterrows():
            #print(workorder)
            if len(self.TRC.loc[((self.TRC['Date'] == workorder['Date']) & (self.TRC['METRAGE'] >= workorder['Start Point']) & (self.TRC['METRAGE'] <= workorder['End Point']))]) == 0:
                print('No track found for {}'.format(workorder['Order']))
            self.TRC.loc[((self.TRC['Date'] == workorder['Date']) & (self.TRC['METRAGE'] >= workorder['Start Point']) & (self.TRC['METRAGE'] <= workorder['End Point'])), 'Work_orders'].apply(lambda x: x.append(workorder['Order']))
            self.TRC.loc[((self.TRC['Date'] == workorder['Date']) & (self.TRC['METRAGE'] >= workorder['Start Point']) & (self.TRC['METRAGE'] <= workorder['End Point'])), 'Work_order_type'].apply(lambda x: x.append(workorder['MAT descriptn']))
        self.TRC['Track_code'] = self.track_code

    def get_track_code(self, x, trunc = False):
        try:
            ans = x[0]
            if trunc == True:
                ans = ans[:3]
        except IndexError: #no track code
            ans = ''
        return ans

    def simplify_track_code(self, df):
        df['Track codes'] = df['Track codes'].apply(lambda x: literal_eval(x))
        df['Track_code_pt1'] = df['Track codes'].apply(lambda x: self.get_track_code(x)[:3])
        df['Track_code_pt2'] = df['Track codes'].apply(lambda x: self.get_track_code(x)[3:6])
        return df

    def match_culvert(self):
        filtered_culvert = self.culvert_df.loc[self.culvert_df['Track_code_pt1'] == self.track_code[:3]]
        print('found {} culverts to match'.format(len(filtered_culvert)))
        self.TRC['culvert'] = 0
        for ind, row in filtered_culvert.iterrows():
            start = row['Start Point']
            end = row['End Point']
            self.TRC.loc[self.TRC['METRAGE'].between(start, end), 'culvert'] = 1

    def match_speed(self):
        filtered_speed = self.speed_df.loc[self.speed_df['Track_code_pt1'] == self.track_code[:3]]
        print('found {} speed rows to match'.format(len(filtered_speed)))
        collist = ['speed_description', 'speed_char_val_from', 'speed_char_val_to', 'speed_value_units']
        for col in collist:
            self.TRC[col] = np.nan
        for ind, row in filtered_speed.iterrows():
            start = row['Start Point']
            end = row['End Point']
            to_add = row[['Description', 'LAM Char. Val. From', 'LAM Char. Val. To', 'MU']].tolist()
            self.TRC.loc[self.TRC['METRAGE'].between(start, end),'speed_description'] = to_add[0]
            self.TRC.loc[self.TRC['METRAGE'].between(start, end),'speed_char_val_from'] = to_add[1]
            self.TRC.loc[self.TRC['METRAGE'].between(start, end),'speed_char_val_to'] = to_add[2]
            self.TRC.loc[self.TRC['METRAGE'].between(start, end),'speed_value_units'] = to_add[3]

    def match_gpr(self):
        filtered_gpr = self.gpr_df.loc[self.gpr_df['Line Segment'] == int(self.track_code[:3])]
        print('found {} gpr rows to match'.format(len(filtered_gpr)))
        collist = filtered_gpr.columns

        '''['Category', 'Category.1', 'Category.2', 'Category.3', 'Category.4',
       'Category.5', 'Category.6', 'Category.7', 'Category.8', 'Centre',
       'Centre.1', 'Centre.2', 'Centre.3', 'Centre.4', 'Centre.5',
       'Collection Date','Division', 'GPR Run Number', 'Laser Run number', 'Left*', 'Left*.1',
       'Left*.2', 'Left*.3', 'Left*.4', 'Left*.5', 'Line Segment', 'Mudspot',
       'PVC Value', 'PVC Value.1', 'PVC Value.2', 'Prefix', 'Right', 'Right*',
       'Right*.1', 'Right*.2', 'Right*.3', 'Right*.4', 'Right.1', 'Right.2',
       'Right.3', 'Sleeper_type', 'Sub-division', 'Track Code',
       'Track ID', 'Volume (cubic m)', 'Volume (cubic m).1',
       'Volume (cubic m).2']'''
        for col in collist:
            self.TRC[col] = np.isnan
        for ind, row in filtered_gpr.iterrows():
            start = row['Start KM']
            end = row['End KM']
            to_add = row[collist].tolist()
            rownum = self.TRC.loc[self.TRC['METRAGE'].between(start,end)].index.tolist()
            self.TRC.loc[rownum, collist] = to_add
            # for i, col in enumerate(collist):
            #     self.TRC.loc[self.TRC['METRAGE'].between(start,end), col] = to_add[i]



    def main_join(self):
        ## ensure work orders in right format
        print('*' * 100)

        self.wo_df['Track_code'] = self.wo_df['Track codes'].apply(lambda x: self.extract_track_code_wo(x))
        ## breakdown track codes in speed and culvert dataset
        self.simplify_track_code(self.culvert_df)
        self.simplify_track_code(self.speed_df)

        ## match TRC to work orders
        print('now joining work orders')
        self.match_work_orders()
        print('joined work orders, checking if successful')
        print('number of rows with work orders')
        print(len(self.TRC[self.TRC['Work_orders'].apply(lambda x: len(x)) >0 ]))
        print('now joining culverts data')
        self.match_culvert()
        #print('number of rows with culverts', len(self.TRC[self.TRC['culvert']==1 ]))

        print('now joining speed data')
        self.match_speed()
        #print(len('number of rows with speed description', self.TRC[self.TRC['speed_description'].isna() ==False ]))
        print('now joining GPR data')
        self.match_gpr()

        return self.TRC

if __name__ == '__main__':
    aligned_df_dict = align_trc(paths.TRC_195).main_align()
    #aligned_df_dict = align_trc(paths.TRC_138, skip = 4).main_align()
    final = []

    wo_df = 'work_orders_with_track_code.csv'
    gps_df = 'C138_C195_coords.csv'
    culvert_df = 'culvert_crossing_with_track_codes.csv'
    speed_df = 'speed_class_with_track_codes.csv'
    gpr_df = 'gpr_195_138.csv'
    for tcode, df in aligned_df_dict.items():
        joined = join_trc(df, wo_df,  gps_df, culvert_df, speed_df, gpr_df, tcode).main_join()
        final.append(joined)

    for ind, data in enumerate(final):
        data.to_csv('trc_joined_{}.csv'.format(ind))

    pd.concat(final).to_csv('combined_195.csv')
