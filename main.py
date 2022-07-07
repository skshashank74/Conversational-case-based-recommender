'''
Travel CCBR
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

os.system('cls' if os.name == 'nt' else 'clear')

class ccbr():
    '''
    class containing the core CCBR functionality
    '''
    def __init__(self, input_file, is_weighted=False) -> None:
        '''
        init method does the following jobs:
        1. parses the input case base file and stores it as a pandas dataframe
        2. initialize variables corresponding to features and questions
        3. does regression to compute feature importance
        '''

        # initialize util object from Util class
        self.util_obj=Util()
        
        # read casebase text file, store it as a pandas dataframe and save it as a csv
        self.df=self.util_obj.build_df(input_file)
        self.df.to_csv('data/travel_cb.csv', encoding='utf-8')
        self.orig_df=pd.read_csv('data/travel_cb_orig.csv', encoding='utf-8')

        # defines if features are to be weighted equally or if the weights are tuned
        self.is_weighted=is_weighted

        # define problem descriptions (features)
        self.predictors=[
            'HolidayType','NumberOfPersons','Region', 'Transportation', 'Duration', 'Season', 
            'Accommodation','Hotel'
        ]

        # calculate range of features to be used for normalization
        self.feature_value_range=self.find_feature_value_range(self.predictors)

        # regression model to compute feature importance -> to be used in generating Questions
        # and also in adapting solution 
        xtrain, xtest, ytrain, ytest = train_test_split(self.df[self.predictors],self.df['Price'], 
                                                  random_state=42, 
                                                  test_size=0.20, shuffle=True)
        self.reg = LinearRegression().fit(xtrain, ytrain)
        self.predictors_1=['NumberOfPersons', 'Duration', 'Season']
        self.reg_1 = LinearRegression().fit(self.df[self.predictors_1], self.df['Price'])
        y_pred_train = self.reg.predict(xtrain)
        y_pred_test = self.reg.predict(xtest)
        coef= np.abs(self.reg.coef_)
        score = coef/sum(coef)
        self.score_dict = {}
        score = list(coef/sum(coef))
        for i in range(len(self.predictors)):
            self.score_dict[self.predictors[i]] = score[i] 
        self.score_dict = {k: v for k, v in sorted(self.score_dict.items(), key=lambda item: item[1],reverse=True)}
        
        # define a rank list for the problem features
        self.running_feature_ranklist=list(self.score_dict.keys())

        # define nominal and ordinal features
        self.nominal_features=['Region', 'Hotel']
        self.ordinal_features=['NumberOfPersons', 'Duration', 'Season']

        # define feature-quesion map
        self.f_q_map={
            'HolidayType':'What holiday type are you looking for? ',
            'NumberOfPersons':'How many people are there in the trip? ',
            'Region':'Which region are you interested in travelling? ',
            'Transportation':'What mode of transportation are you looking for? ',
            'Duration':'What is the duration of the trip? ',
            'Season':'In which Season are you planning the trip? ',
            'Accommodation':'What type of accomodation you want?',
            'Hotel':'Do you have any preference of Hotel?'
        }

        # detailed questions to imptove UX
        self.f_q_map_detailed={
            'HolidayType':('What holiday type are you looking for?\n'
                'Choose 1 for Active\nChoose 2 for Bathing\nChoose 3 for City\n'
                'Choose 4 for Education\nChoose 5 for Language\nChoose 6 for Recreation\n'
                'Choose 7 for Skiing\nChoose 8 for Wandering\n'),
            'NumberOfPersons':'How many people are there in the trip? \n',
            'Region':'Which region are you interested in travelling to? \n',
            'Transportation':('What mode of transportation are you looking for?\n'
                'Choose 1 for Car\nChoose 2 for Coach\nChoose 3 for Plane\n'
                'Choose 4 for Train\n'),
            'Duration':'What is the duration of the trip?\n',
            'Season':('In which Season are you planning the trip?\n'
                'Choose 1 for January, 2 for February ... 12 for December\n'),
            'Accommodation':('What type of accomodation you want?\n'
                'Choose 1 for HolidayFlat\nChoose 2 for OneStar\nChoose 3 for TwoStars\n'
                'Choose 4 for ThreeStars\nChoose 5 for FourStars\nChoose 6 for FiveStars\n'),
            'Hotel':'Do you have any preference of Hotel?\n'
        }

        # define function to store user responses
        self.user_pref={}

    def start(self):
        '''
        method that outlines the flow of the program
        1. Print out Questions to the user
        2. Use responses from the user to find cases
        3. Present cases to the user and ask the user if they would like to select a case or not
        4. If user selects a case->print out the selected case
        5. Else present additional questions to the user
        6. Repeat steps 3 to 5 until the features are exhausted or user selects a case
        7. Adapt the case as per user preference and print out the solution (price)
        '''
        print('Welcome to Travel CBR system!')
        print()
        print('is_weighted: ', self.is_weighted)
        print()

        # while loop that runs until the 1. user selects a case or 2. all the questions 
        # (corresponding to problem features) are exhausted 
        while self.running_feature_ranklist:
            # Present questions to the user and get their response
            selected_feature=self.get_q_preference()

            # Convert the question selected by the user and their response to standardized 
            # format (integers) to be stored in pandas dataframe
            selected_feature_val=self.get_feature_val(selected_feature)

            # build a dictionary to store responses from the user
            self.user_pref[selected_feature]=\
                self.convert_to_ordinal(selected_feature, selected_feature_val)
            print()
            print('User selection (running): ')
            print(self.user_pref)
            print()
            
            # retrieve similar cases
            # send similarity_metric and is_weighted parmaeters to the function call
            best_case_ids, best_case_scores=self.get_similar_cases(
                self.user_pref, weights=self.is_weighted
            )
            best_case_scores=abs(best_case_scores)

            # check if user wants to select a case
            is_final=self.check_case_with_user(best_case_ids, best_case_scores)
            if is_final:
                # print final result if user selects a case
                self.print_final_price()
                break
            else:
                pass
        # adaptation logic if user doesn't select a case
        if not is_final:
            print('User selection (Final): ')
            print(self.user_pref)
            print()
            self.print_final_price(self.adapt_case(self.user_pref))
        
    def get_q_preference(self):
        '''
        Print top 3 questions
        Ask user to select 1 of the questions
        Store user selection
        Ask user to answer the selected question
        Store answer
        '''
        q_scores=[self.score_dict.get(feature) for feature in self.running_feature_ranklist][:3]
        q_scores/=sum(q_scores)
        for i in range(3):
            try:
                ques=self.f_q_map.get(self.running_feature_ranklist[i])
            except IndexError:
                break
            print(f'{i+1}: {ques} (Score: {round(q_scores[i],2)})')
        selected_feature_id=None
        while not selected_feature_id:
            try:
                print()
                temp=int(input('Select one of the above Qs (1, 2 or 3): \n'))
                if temp in range(1, len(self.running_feature_ranklist)+1):
                    selected_feature_id=temp
                else:
                    print('Select a valid Question number! Try again!')
            except ValueError:
                print('Select a valid Question number! Try again!')
        selected_feature_name=self.running_feature_ranklist[selected_feature_id-1]
        self.running_feature_ranklist.pop(selected_feature_id-1)
        return selected_feature_name

    def get_feature_val(self, selected_feature):
        '''
         Store user's answer
        '''
        if selected_feature not in self.nominal_features:
            print('\n')
            return int(input(self.f_q_map_detailed.get(selected_feature)))
        else:
            print('\n')
            return input(self.f_q_map.get(selected_feature))
    
    def get_similar_cases(self, user_pref, k=3, weights=False):
        '''
        initialize weights_arr based on boolean weights variable
        if weights is set to False: use equal weights to calculate similarity
        if weights is set to True: use weights calculated using regression
        use McSherry formula to calculate similarity between individual features
        then use the l2 norm to calculate the euc distance to calculate the total similarity
        '''
        weights_arr=np.ones(len(self.predictors))/len(self.predictors)
        if weights:
            weights_arr=np.asarray(list(self.score_dict.values()))
        # retrieve indices of k similar cases
        euc_dist_arr=np.full(len(self.df), np.inf).reshape(-1,1)
        for count, val in enumerate(euc_dist_arr):
            temp_similarity_arr=[0]*len(self.predictors)
            for count_i, val_i in enumerate(self.predictors):
                temp_similarity_arr[count_i]=self.find_feature_similarity(val_i, user_pref.get(val_i), self.df.at[count,val_i])
                temp_similarity_arr[count_i]*=weights_arr
            euc_dist_arr[count]=np.linalg.norm(temp_similarity_arr)
        euc_dist_arr=np.amax(euc_dist_arr, axis=1)
        return np.argsort(-euc_dist_arr)[:k], np.sort(-euc_dist_arr)[:k]

    def find_feature_similarity(self, feature_name, q_feature, c_feature):
        '''
        calculate similarity between two features
        for nominal features-> return 1 if same, 0 otherwise
        for ordinal features-> return 1-(normalized distance between features) (McSherry 2014)
        '''
        if not q_feature:
            return 0
        elif feature_name not in self.ordinal_features:
            return 1 if q_feature==c_feature else 0
        elif feature_name=='Season':
            return 1 if q_feature==c_feature else 0
        else:
            return 1-(abs(q_feature-c_feature)/self.feature_value_range.get(feature_name))
    
    def find_feature_value_range(self, predictors):
        '''
        returns range of a feature wrt all the entries in the casebase
        '''
        feature_value_range={key: 0 for key in predictors}
        for feature in predictors:
            feature_value_range[feature]=self.df[feature].max()-self.df[feature].min()
        return feature_value_range

    def check_case_with_user(self, best_case_ids, best_case_scores):
        '''
        Ask user if they would like to select a journey
        '''
        best_case_scores/=sum(best_case_scores)
        for index, (val1, val2) in enumerate(zip(best_case_ids, best_case_scores)):
            print('\n')
            print('Case Score: ', round(val2,2))
            print(f'{self.orig_df.iloc[val1, [0,1,3,4,5,6,7,8,9]].to_string()}')
        print('\n')
        is_final=input('Do you want to select a Journey? [y/n]\n')
        print()
        if is_final=='y':
            return True
        else:
            return False
    
    def print_final_price(self, price=None):
        '''
        Print final result
        '''
        if not price:
            selected_journey_code=int(input('Select Journey Code: '))
            print()
            final_price=self.df.iloc[selected_journey_code-1]['Price']
            print('Here is your selected journey:')
            print('########################')
            print(f'{self.orig_df.iloc[selected_journey_code-1, [0,1,3,4,5,6,7,8,9]].to_string()}')
            print('########################')
            print(f'Final price is: {final_price}')
            print('########################')
            print()
            print('Thanks for visiting Travel CBR system!')  
            print()
        else:
            best_case_id, best_case_score=self.get_similar_cases(
                self.user_pref, metric=self.similarity_metric, weights=self.is_weighted
            )
            print('Here\'s the best match:')
            print('########################')
            print(f'{self.orig_df.iloc[best_case_id[0], [1,3,4,5,6,7,8,9,2]].to_string()}')
            print('########################')
            print(f'Based on the input, the Case-adapted Price is: {round(price,2)}')   
            print('########################')  
            print()
            print('Thanks for visiting Travel CBR system!')  
            print()  

    def convert_to_ordinal(self, feature, nom_val):
        '''
        convert all features to integers
        '''
        if feature=='Region':
            return self.util_obj.regions.get(nom_val)
        elif feature=='Hotel':
            return self.util_obj.hotels.get(nom_val)
        else:
            return int(nom_val)
    
    def adapt_case(self, user_pref):
        '''
        adapt case using coefficients calculated through regression
        '''
        user_pref_list=[None]*len(self.predictors)
        for k,v in user_pref.items():
            user_pref_list[self.predictors.index(k)]=v
        return self.reg_1.predict([list(user_pref_list[i] for i in [1, 4, 5] )])[0]

class Util():
    '''
    Util class
    '''
    def __init__(self):
        self.features = {'JourneyCode':0, 'HolidayType':1, 'Price':2, 'NumberOfPersons':3, 'Region':4,
                'Transportation':5, 'Duration':6, 'Season':7, 'Accommodation':8, 'Hotel':9}
    
    def build_df(self, input_file):
        with open(input_file, 'r+') as f:
            lines = f.readlines()
        travel_cb_dict={}
        for i in range(len(lines)):
                if self.is_case_object(lines[i]):
                    case_key=self.get_case_key(lines[i])
                    feature_list=[None]*len(self.features)
                    i+=1
                    while self.is_feature(lines[i]):
                        feature_index=self.features[self.get_feature_key(lines[i])]
                        feature_value=self.get_feature_value(lines[i], self.get_feature_key(lines[i]))
                        feature_list[feature_index]=feature_value
                        i+=1
                    travel_cb_dict[case_key]=feature_list
        df = pd.DataFrame(travel_cb_dict.values(), columns = self.features.keys())
        df.to_csv('data/travel_cb_orig.csv', encoding='utf-8', index=False)

        self.holiday_types = {k: v+1 for v, k in enumerate(sorted(df['HolidayType'].unique()))}
        self.regions={k: v+1 for v, k in enumerate(sorted(df['Region'].unique()))}
        self.transportation_modes={k: v+1 for v, k in enumerate(sorted(df['Transportation'].unique()))}
        self.seasons={k: v+1 for v, k in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
                'December'])}
        self.accomodations={k: v+1 for v, k in enumerate(['HolidayFlat', 'OneStar', 'TwoStars', 'ThreeStars', 'FourStars', 'FiveStars'])}
        self.hotels={k: v+1 for v, k in enumerate(df['Hotel'].unique())}

        travel_cb_dict_new={}
        for k,v in travel_cb_dict.items():
            travel_cb_dict_new[k]=self.standardize_feature_list(v)
        df = pd.DataFrame (travel_cb_dict_new.values(), columns = self.features.keys())
        return df.astype(int)

    def is_case_object(self, input_line)->bool:
        '''checks if input line is a case object'''
        return bool(len(input_line)-len(input_line.lstrip('\t'))==2
                    and input_line.strip().split()[0]=='case')

    def is_feature(self, input_line)->bool:
        '''checks if input line is a feature element'''
        return bool(len(input_line)-len(input_line.lstrip('\t'))==3
                    and input_line.strip().split()[0].strip(':') in self.features)
    
    def get_case_key(self, input_line):
        return input_line.strip().split()[1]

    def get_feature_key(self, input_line):
        return input_line.strip().split()[0].strip(':')

    def get_feature_value(self, input_line, featurekey):
        return input_line.strip().removeprefix(featurekey).strip(':,. "')
    
    def standardize_feature_list(self, featurelist):
        featurelist[1]=self.holiday_types[featurelist[1]]
        featurelist[4]=self.regions[featurelist[4]]
        featurelist[5]=self.transportation_modes[featurelist[5]]
        featurelist[7]=self.seasons[featurelist[7]]
        featurelist[8]=self.accomodations[featurelist[8]]
        featurelist[9]=self.hotels[featurelist[9]]
        return featurelist

if __name__=='__main__':
    input_file='data/travel_cb.txt'
    travel_cbr=ccbr(input_file, is_weighted=False)
    # travel_cbr.start()
    travel_cbr.start()