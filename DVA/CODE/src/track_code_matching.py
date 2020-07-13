import pandas as pd
import numpy as np



class Translate_Location():
    def __init__(self, code_ref_step1, code_ref_step2, code_ref_step2a, code_ref, df, index = "rownum"):
        self.code_ref_step1 = self.col_to_list(code_ref_step1) # 4th level LRP
        self.code_ref_step2 = self.col_to_list(code_ref_step2) # 5th level LRP
        self.code_ref_step2a = code_ref_step2a #LRP details
        self.code_ref = code_ref #Track code list
        self.df = df #df containing all work orders/ column with functional location
        self.ind = index # column to use as index

    def col_to_list(self, df_to_compress):
        ## returns new df with columns compressed to list with LRPID as key
        new_dict = dict()
        for ind, row in enumerate(df_to_compress.iterrows()):
            a = row[1][1:].dropna().tolist()
            try:
                new_dict[ind] = [row[1][0], [int(x) for x in a]]
            except ValueError: ## catch NaN
                new_dict[ind] = [row[1][0], [x for x in a]]
        return pd.DataFrame.from_dict(new_dict, orient='index', columns=['LRPID', 'ref_num'])

    def step_1(self, level):
        # step one, get candidate track codes from 4th level
        first_3_digits = self.code_ref_step1[self.code_ref_step1['LRPID'] == level[3]]['ref_num'].to_list()[0]

        track_codes = []
        for d in first_3_digits:
            d_str = str(d)
            t = self.code_ref[self.code_ref['Track code'].str.startswith(d_str)]['Track code'].to_list()
            track_codes.extend(t)
        return track_codes

    def step_2(self, level, possible_nums):
        # step two, filter possible track codes using 5th level
        ## check if results have both possible track codes based on 5th level --> step 2A if true
        level4 = level[4]
        digit4 = [x[3] for x in possible_nums]
        ans = [] # to filter
        try:
            possible_4th_num = self.code_ref_step2[self.code_ref_step2['LRPID'] == level4]['ref_num'].to_list()[0]
        except IndexError:
            ## not in list
            return possible_nums, False # return same list

        num_dict = {str(i):0 for i in possible_4th_num} #initialise for counting occurrances
        for x in digit4:
            if str(x) in str(possible_4th_num):
                ans.append(True)
                num_dict[str(x)] +=1
            else:
                ans.append(False)
        ans2 = True
        for num, count in num_dict.items():
            if count > 0:
                ans2 = ans2 and True
            else:
                ans2 = False
        return np.array(possible_nums)[np.array(ans)], ans2

    def step_2A(self, level, possible_nums):
    ## filter again to include those in LRP details
    ## not sure what to do after
        self.code_ref_step2a[self.code_ref_step2a['LRPID'].str.startswith('C138')]['LRPID'].map(lambda x: x.split('-')[-1])

    def step_3(self, work_loc_start, work_loc_end, possible_nums):
        # filter based on location of works
        ans = []
        for i in possible_nums: #loop through all possible numbers
            track_s = self.code_ref[self.code_ref['Track code'] == i]['Start KM'].to_list()
            track_e = self.code_ref[self.code_ref['Track code'] == i]['End KM'].to_list()
            track_length = track_s + track_e ## added this as it could be bi-directional
            track_start = min(track_length)
            track_end = max(track_length)
            work = [float(str(work_loc_start).replace(',','')), float(str(work_loc_end).replace(',',''))]
            work_start = min(work)
            work_end = max(work)
            # check if work location is more or less than track start or end
            if track_start> work_start:
                ans.append(False)
            elif track_end < work_end:
                ans.append(False)
            else:
                ans.append(True)
        return np.array(possible_nums)[np.array(ans)]

    def funcloc_to_trackcode(self):
        ## main function to return track codes
        ans = dict()
        for index, row in enumerate(self.df.iterrows()):
            if self.ind == 'rownum':
                order = index
            else:
                order = row[1][self.ind] # indexed by
            func_loc = row[1]['Functional Loc.']
            start = row[1]['Start Point']
            end = row[1]['End Point']
            try:
                level = str(func_loc).split('-')
            except AttributeError: #NA
                level = [func_loc]
            level_len = len(level)
            if level_len == 4:
                ## track code only has 4 levels
                # we can only match level 4 with start and end location info
                s3 = self.step_3(start, end, self.step_1(level))
                ans[index] = [order, s3.tolist()]
            elif level_len > 4:
                ## can proceed to step 2 and 3
                # step1
                s1 = self.step_1(level)
                # step2
                s2, if_both = self.step_2(level, s1)
                if if_both == True:
                    ## step 2A I don't quite understand this one
                    pass #for now
                ## step 3, filter to only track codes within range
                s3 = self.step_3(start, end, s2)
                ans[index] = [order, s3.tolist()]
            else: ## no matching code
                ans[index] = [order, []]
        return pd.DataFrame.from_dict(ans, orient='index', columns = [self.ind, 'Track codes'])




if __name__ == '__main__':
    from utils import paths
    import os
    ## read all data files
    code_ref_4th = pd.ExcelFile(os.path.join(paths.DATA_OCT, '4th Level LRP to 3 Digit Track Code.xlsx'))
    code_ref_step1= pd.read_excel(code_ref_4th, 'Matrix')
    code_ref_step2 =  pd.read_excel(os.path.join(paths.DATA_OCT, '5th Level LRP to 4th Digit Track Code.xlsx.xlsx')).iloc[:7,:]
    code_ref_step2a = pd.read_excel(os.path.join(paths.DATA_OCT, 'LRP Details.xlsx'))
    code_ref = pd.ExcelFile(os.path.join(paths.DATA_OCT, 'Track Code List_2.xlsx'))
    code_ref = pd.read_excel(code_ref)
    code_ref['Track code'] = code_ref['Track code'].map(lambda x: str(x))
    work_order = pd.read_excel(os.path.join(paths.DATA_OCT, 'Work Orders for Corridors C138 and C195 10.10.2019 including costs.xlsx'))


    ## init class
    trans = Translate_Location(code_ref_step1, code_ref_step2, code_ref_step2a, code_ref, work_order, 'Order')
    track_code_translated = trans.funcloc_to_trackcode()
    track_code_translated.to_csv('track_code_translated.csv', index = False )
