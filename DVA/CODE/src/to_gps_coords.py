import pandas as pd
import numpy as np
from ast import literal_eval



class Match_gps():
    def __init__(self, gps_df, df_to_match, track_code = None, track_funcloc = None):
        ## match dataset using the first 3 numbers in track code + KM information to gps
        self.gps_df = gps_df
        self.df_to_match = df_to_match
        self.track_code = track_code # for defined track code in TRC data, should be a string
        self.track_funcloc = track_funcloc ## if we just want to force match with level4 func location
        self.work_col = ['Order', 'Track_code', 'Work_type','Start_lat', 'Start_long', 'End_lat', 'End_long']
        self.cul_col = ['culvert_loc', 'Track_code', 'Start_lat', 'Start_long', 'End_lat', 'End_long']



    def get_gps(self, trackID, start_value, end_value):
        ## match dataset using the first 3 numbers in track code + KM information to gps
        ### returns gps information of start and end
        if trackID != '':
            first3 = int(str(trackID)[:3])
            last3 = int(str(trackID)[3:])
            segmented = self.gps_df[self.gps_df['Line Segment'] == first3] ## match line segment
        else: # force to level 4 functional location
            segmented = self.gps_df[self.gps_df['track_funcloc'] == self.track_funcloc]

        if start_value != end_value:
            nearest_ind_start = segmented.iloc[(segmented['Start KM']-start_value).abs().argsort()[:1]]
            nearest_ind_end = segmented.iloc[(segmented['Start KM']-end_value).abs().argsort()[:1]]
            try:
                nearest_lat_start = nearest_ind_start['Dec.Lat'].values[0]
                nearest_long_start = nearest_ind_start['Dec.Long'].values[0]
                nearest_lat_end = nearest_ind_end['Dec.Lat'].values[0]
                nearest_long_end = nearest_ind_end['Dec.Long'].values[0]
                return nearest_lat_start, nearest_long_start, nearest_lat_end, nearest_long_end
            except IndexError: #cannot find closest value
                print(trackID, start_value, end_value)
                return np.nan,np.nan,np.nan, np.nan
        else:
            nearest_ind_start = segmented.iloc[(segmented['Start KM']-start_value).abs().argsort()[:1]]
            try:
                nearest_lat = nearest_ind_start['Dec.Lat'].values[0]
                nearest_long = nearest_ind_start['Dec.Long'].values[0]
                return  nearest_lat, nearest_long, nearest_lat, nearest_long
            except IndexError: #cannot find closest value
                print(trackID, start_value, end_value)
                return np.nan,np.nan,np.nan, np.nan

    def work_order_to_gps(self):
        ## method to match gps data to work order
        ans = dict()
        for index, row in self.df_to_match.iterrows():
            order = row['Order']
            codes = literal_eval(row['Track codes'])
            start = float(str(row['Start Point']).replace(',', ''))
            end = float(str(row['End Point']).replace(',',''))
            work_type = row['MAT descriptn']
            for code in codes:
                start_gps_lat, start_gps_long, end_gps_lat, end_gps_long = self.get_gps(code, start, end)
                ans[(order, code[0], work_type)] = [start_gps_lat, start_gps_long,end_gps_lat,end_gps_long]

        multi_index = pd.MultiIndex.from_tuples(ans.keys())
        df = pd.DataFrame(list(ans.values()), index = multi_index).reset_index()
        df.columns = self.work_col
        return df

    def culvert_to_gps(self):
        ## method to match gps data to work order
        ans = dict()
        for index, row in self.df_to_match.iterrows():
            cul_loc = row['culvert_loc']
            codes = literal_eval(row['Track codes'])
            start = float(str(row['Start Point']).replace(',', ''))
            end = float(str(row['End Point']).replace(',',''))
            try:
                start_gps_lat, start_gps_long, end_gps_lat, end_gps_long = self.get_gps(codes[0], start, end)
                ans[(cul_loc, codes[0])] = [start_gps_lat, start_gps_long,end_gps_lat,end_gps_long]
            except IndexError:#no track code
                ans[(cul_loc, np.nan)] = [np.nan, np.nan, np.nan, np.nan]
        multi_index = pd.MultiIndex.from_tuples(ans.keys())
        df = pd.DataFrame(list(ans.values()), index = multi_index).reset_index()
        df.columns = self.cul_col
        return df

    def exceptions_to_gps(self):
        ## method to match gps data to work order
        ans = dict()
        for index, row in self.df_to_match.iterrows():
            ind = row['rownum']
            code = row['TrackCode']
            loc = float(str(row['Location']).replace(',', ''))
            try:
                start_gps_lat, start_gps_long, end_gps_lat, end_gps_long = self.get_gps(code, loc, loc)
                ans[(ind, code)] = [start_gps_lat, start_gps_long]
            except IndexError:#no track code
                ans[(cul_loc, np.nan)] = [np.nan, np.nan, np.nan, np.nan]
        multi_index = pd.MultiIndex.from_tuples(ans.keys())
        df = pd.DataFrame(list(ans.values()), index = multi_index).reset_index()
        df.columns = ['rownum', 'TrackCode', 'latitude','lonitude']
        return df

    def speed_to_gps(self):
        ans = dict()
        for index, row in self.df_to_match.iterrows():
            codes = literal_eval(row['Track codes'])
            start = float(str(row['Start Point']).replace(',', ''))
            end = float(str(row['End Point']).replace(',',''))
            if len(codes)== 0:
                start_gps_lat, start_gps_long, end_gps_lat, end_gps_long = self.get_gps('', start, end)
                ans[(index, '')] = [start_gps_lat, start_gps_long,end_gps_lat,end_gps_long]
            else:
                for code in codes:
                    start_gps_lat, start_gps_long, end_gps_lat, end_gps_long = self.get_gps(code, start, end)
                    ans[(index, code)] = [start_gps_lat, start_gps_long,end_gps_lat,end_gps_long]
        multi_index = pd.MultiIndex.from_tuples(ans.keys())
        df = pd.DataFrame(list(ans.values()), index = multi_index).reset_index()
        df.columns = ['rownum', 'Track_code','Start_lat', 'Start_long', 'End_lat', 'End_long']
        return df

    def TRC_to_gps(self):
        ## method to match gps data to TRC data
        ## needs to interpolate, since TRC (every 1m) is more granular than GPR1 (every 5m) data
        if self.track_code == None:
            raise Exception('Please enter track code for TRC data')
        elif len(str(self.track_code)) == 3:
            self.track_code = str(self.track_code)
            Line_segment = self.track_code[:3]
            self.gps_df['Line Segment']  = self.gps_df['Line Segment'].apply(lambda x: str(x))
            filteredf = self.gps_df[(self.gps_df['Line Segment'] == Line_segment)]
            if filteredf.shape[0]==0:
                print(Line_segment, Track_ID)
                print(filteredf.shape)
                raise Exception('Track Code not found in GPS data, please check')

            else:
                long = filteredf['Dec.Long'].values
                lat = filteredf['Dec.Lat'].values
                dist = filteredf['Start KM'].values
                self.df_to_match['Latitude'] = self.df_to_match['METRAGE'].apply(lambda x: np.interp(x, dist, lat))
                self.df_to_match['Longitude'] = self.df_to_match['METRAGE'].apply(lambda x: np.interp(x, dist, long))
        else:
            self.track_code = str(self.track_code)
            Line_segment = self.track_code[:3]
            Track_ID = self.track_code[3:]
            self.gps_df['Line Segment']  = self.gps_df['Line Segment'].apply(lambda x: str(x))
            self.gps_df['Track ID']  = self.gps_df['Track ID'].apply(lambda x: str(x))

            filteredf = self.gps_df[(self.gps_df['Line Segment'] == Line_segment) & (self.gps_df['Track ID'] == Track_ID)]
            if filteredf.shape[0]==0:
                print(Line_segment, Track_ID)
                print(filteredf.shape)
                raise Exception('Track Code not found in GPS data, please check')

            else:
                long = filteredf['Dec.Long'].values
                lat = filteredf['Dec.Lat'].values
                dist = filteredf['Start KM'].values
                self.df_to_match['Latitude'] = self.df_to_match['METRAGE'].apply(lambda x: np.interp(x, dist, lat))
                self.df_to_match['Longitude'] = self.df_to_match['METRAGE'].apply(lambda x: np.interp(x, dist, long))

        return self.df_to_match


if __name__ == '__main__':
    from utils import paths
    import os
 # Example for work order data

    # gps_df = pd.read_csv(os.path.join(paths.DATA_DERIVED, 'C138_C195_coords.csv'))
    # work_orders = pd.read_csv(os.path.join(paths.DATA_DERIVED,'work_order_with_track_code.csv'))
    #
    # output = Match_gps(gps_df, work_orders).work_order_to_gps()
    # output.to_csv(os.path.join(paths.DATA_DERIVED, 'work_orders_with_coords.csv'), index = False)

# Example for TRC data
    for file in os.listdir(paths.TRC):
        split_name = file.split(' ')
        outname = 'TRC_{0}_coords.csv'.format(split_name[-1].split('.')[0])

        gps_data = pd.read_csv(os.path.join(paths.DATA_DERIVED, 'gps_data.csv'))
        TRC_data = pd.read_csv(os.path.join(paths.TRC, file), skiprows = 4)

        output = Match_gps(gps_data, TRC_data, track_code= split_name[0]).TRC_to_gps()

        output.to_csv(os.path.join(paths.DATA_DERIVED, outname))
