#!/usr/bin/env python
# coding: utf-8

# # Corporate PAC Inclusion Analysis¶
# ## Does your company's Political Action Committee have a blind spot with respect to inclusion?

# In[1]:


# Execute this cell to load the notebook's style sheet, then ignore it
from IPython.core.display import HTML
css_file = 'PAC.css'
HTML(open(css_file, "r").read())


# In[2]:


import pandas as pd
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import Javascript, display, HTML


# In[3]:


races = ['Asian', 'Black', 'Hispanic', 'Other Races', 'White']
genders = ['Male', 'Female']


# In[4]:


pac_selection = widgets.Select(
    options=['BlackRock', 'Leidos', 'Google'],
    value='BlackRock',
    description='Select PAC:',
    disabled=False,
)

pac_text = widgets.Text(value="BlackRock")
def handle_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        print("changed to %s" % change['new'])
        pac_text.value = change['new']
        display(Javascript('IPython.notebook.execute_cells_below()'))

pac_selection.observe(handle_change)
display(pac_selection) 


# In[5]:


def display_race_charts(input_df, main_title):
    #strip Total column from dataframe
    df = input_df.iloc[:,:-1]
    count = len(df.index)
    fig, axes = plt.subplots(1,count, figsize=(12,3))
    if count > 1:
        for ax, idx in zip(axes, df.index):
            ax.pie(df.loc[idx], labels=df.columns, radius=1, autopct='%1.1f%%', explode=(0, 0, 0, 0, 0.1), shadow=True)
            ax.set(ylabel='', title=idx, aspect='equal')
    else:
        ax = df.iloc[0].plot(kind = 'pie', labels=df.columns, radius=1, autopct='%1.1f%%', explode=(0,0,0,0,0.1), shadow=True)
        ax.set(ylabel='', title=df.index[0], aspect='equal')        
    fig.suptitle(main_title, fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    plt.show()

def display_race_summary_charts(input_df, main_title):
    df = input_df
    count = len(df.index)
    fig, axes = plt.subplots(1,count, figsize=(12,3))
    df.plot(kind = 'pie', radius=1, autopct='%1.1f%%', ax=ax, subplots=True, shadow=True)
    ax.set(ylabel='', title=df.index[0], aspect='equal')
    #ax.get_legend().remove()
    fig.suptitle(main_title, fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    plt.show()
    
def display_gender_charts(input_df, main_title):
    #strip Total column from dataframe
    df = input_df.iloc[:,:-1]
    count = len(df.index)
    fig, axes = plt.subplots(1,count, figsize=(12,3))
    if count > 1:
        for ax, idx in (zip(axes, df.index)):
            ax.pie(df.loc[idx], labels=df.columns, radius=1, autopct='%1.1f%%', explode=(0.1, 0), shadow=True)
            ax.set(ylabel='', title=idx, aspect='equal')
    else:
        ax = df.iloc[0].plot(kind = 'pie', labels=df.columns, radius=1, autopct='%1.1f%%', explode=(0.1, 0), shadow=True)
        ax.set(ylabel='', title=df.index[0], aspect='equal')
    fig.suptitle(main_title, fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    plt.show()

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '${v:d}'.format(v=val)
    return my_format

def display_money_charts(input_df, main_title):
    #strip Total column from dataframe
    df = input_df.iloc[:,:-1]
    #Reverse order dataframe
    df = df.iloc[::-1]
    fig, axes = plt.subplots(1,2, figsize=(12,3))
    for ax, idx in zip(axes, df.index):
        ax.pie(df.loc[idx], labels=df.columns, radius=1, autopct = autopct_format(df.loc[idx]), shadow=True)
        ax.set(ylabel='', title=idx, aspect='equal')

    fig.suptitle(main_title, fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    plt.show()

def display_money_bar(input_df, main_title):
    #strip Total column from dataframe
    df = input_df.iloc[:,:-1]
    #Reverse order dataframe
    #df = df.iloc[::-1]
    ax = df.plot(kind='bar', title =main_title, figsize=(15, 10), legend=True, fontsize=12)
    ax.set_xlabel("Party", fontsize=12)
    ax.set_ylabel("PAC Disbursment Dollars", fontsize=12)
    for p in ax.patches:
        ax.annotate('$'+str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()
    
def change_index_values(df, old, new):
    as_list = df.index.tolist()
    idx = as_list.index(old)
    as_list[idx] = new
    df.index = as_list
    
def currency(x, pos):
    'The two args are the value and tick position'
    if x >= 1000000:
        return '${:1.1f}M'.format(x*1e-6)
    return '${:1.0f}K'.format(x*1e-3)

def display_bar(df):
    fig, ax = plt.subplots()
    df.plot(kind = 'barh', ax=ax)
    fmt = FuncFormatter(currency)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.grid()
    
def race_totals(df):
    total = 0
    for r in races:
        if r in df.columns:
            total = total + df[r]
    return total

def gender_totals(df):
    total = 0
    for r in genders:
        if r in df.columns:
            total = total + df[r]
    return total


# In[6]:


congress116_df = pd.read_csv("../data/116th_congress_190103.csv")

#data cleaning

#split race & ethnicity
congress116_df[['Race','Ethnicity']] = congress116_df.raceEthnicity.str.split(" - ", 1, expand=True,)

#remove independents
congress116_df = congress116_df[congress116_df.party != "Independent"]

#remove non-voting members
congress116_df['raceEthnicity'].replace('', np.nan, inplace=True)
congress116_df.dropna(subset=['raceEthnicity'], inplace=True)

#Fix Golden	Jared missing gender
congress116_df.at[240, 'gender'] = 'M'

#change Native American and Pacific Islander to "Other Races"
congress116_df.loc[(congress116_df.Race == 'Native American'),'Race']='Other Races'
congress116_df.loc[(congress116_df.Race == 'Pacific Islander'),'Race']='Other Races'

#change M & F to Male and Female
congress116_df.loc[(congress116_df.gender == 'M'),'gender']='Male'
congress116_df.loc[(congress116_df.gender == 'F'),'gender']='Female'

#remove nan
congress116_df = congress116_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))

#convert floats to ints
#congress116_df = congress116_df.convert_dtypes()


# In[7]:


race_df = congress116_df.groupby(['party','Race']).agg('size').unstack()
gender_df = congress116_df.groupby(['party','gender']).agg('size').unstack()

#remove NaN
race_df = race_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
gender_df = gender_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))

#add total column
race_df['Total'] = race_totals(race_df)
gender_df['Total'] = gender_totals(gender_df)
#race_df['Total'] = race_df['Asian'] + race_df['Black'] + race_df['Hispanic'] + race_df['Other Races'] + race_df['White']
#gender_df['Total'] = gender_df['Male'] + gender_df['Female']

#load 2017 US race and gender data
us_race_df = pd.read_csv("../data/us_race_2017.csv")
us_race_df.set_index('Category', inplace=True)
us_gender_df = pd.read_csv("../data/us_gender_2017.csv")
us_gender_df.set_index('Category', inplace=True)

#concatenate race dataframes
race_df = pd.concat([race_df, us_race_df])
#concatenate gender dataframes
gender_df = pd.concat([gender_df, us_gender_df])

#convert floats to ints
race_df = race_df.convert_dtypes()
gender_df = gender_df.convert_dtypes()

#change index values for clearer presentation
change_index_values(race_df, "Democrat", "Democrats")
change_index_values(race_df, "Republican", "Republicans")
change_index_values(gender_df, "Democrat", "Democrats")
change_index_values(gender_df, "Republican", "Republican")


# In[8]:


#align PAC data
PAC_2017_2018_df = pd.read_csv("../data/"+pac_text.value+"PAC-2017-2018-disbursements.csv")
ThunderboltPAC_2017_2018_df = pd.read_csv("../data/thunderboltPAC-2017-2018-disbursements.csv")
NewPAC_2017_2018_df = pd.read_csv("../data/newPAC-2017-2018-disbursements.csv")

#remove nan
PAC_2017_2018_df = PAC_2017_2018_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
ThunderboltPAC_2017_2018_df = ThunderboltPAC_2017_2018_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
NewPAC_2017_2018_df = NewPAC_2017_2018_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))

#convert floats to ints
PAC_2017_2018_df.candidate_office_district = PAC_2017_2018_df.candidate_office_district.astype(int)
ThunderboltPAC_2017_2018_df.candidate_office_district = ThunderboltPAC_2017_2018_df.candidate_office_district.astype(int)
NewPAC_2017_2018_df.candidate_office_district = NewPAC_2017_2018_df.candidate_office_district.astype(int)

#convert join columns from all caps to capitalized
PAC_2017_2018_df.candidate_first_name = PAC_2017_2018_df.candidate_first_name.str.title()
PAC_2017_2018_df.candidate_last_name = PAC_2017_2018_df.candidate_last_name.str.title()
PAC_2017_2018_df.candidate_middle_name = PAC_2017_2018_df.candidate_middle_name.str.title()
ThunderboltPAC_2017_2018_df.candidate_first_name = ThunderboltPAC_2017_2018_df.candidate_first_name.str.title()
ThunderboltPAC_2017_2018_df.candidate_last_name = ThunderboltPAC_2017_2018_df.candidate_last_name.str.title()
ThunderboltPAC_2017_2018_df.candidate_middle_name = ThunderboltPAC_2017_2018_df.candidate_middle_name.str.title()
NewPAC_2017_2018_df.candidate_first_name = NewPAC_2017_2018_df.candidate_first_name.str.title()
NewPAC_2017_2018_df.candidate_last_name = NewPAC_2017_2018_df.candidate_last_name.str.title()
NewPAC_2017_2018_df.candidate_middle_name = NewPAC_2017_2018_df.candidate_middle_name.str.title()

#join PAC and congress demographic dataframes
PAC_merge_df = pd.merge(PAC_2017_2018_df, congress116_df,  how='left', left_on=['candidate_last_name','candidate_office_district','candidate_office_state'], right_on = ['lastName','district','state'])
TBPAC_merge_df = pd.merge(ThunderboltPAC_2017_2018_df, congress116_df,  how='left', left_on=['candidate_last_name','candidate_office_district','candidate_office_state'], right_on = ['lastName','district','state'])
NewPAC_merge_df = pd.merge(NewPAC_2017_2018_df, congress116_df,  how='left', left_on=['candidate_last_name','candidate_office_district','candidate_office_state'], right_on = ['lastName','district','state'])

#remove nan
PAC_merge_df = PAC_merge_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
TBPAC_merge_df = TBPAC_merge_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
NewPAC_merge_df = NewPAC_merge_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))


# In[9]:


#subsets
#possible loser candidates
winners_2018_df = PAC_merge_df.loc[PAC_merge_df['Race'] != ""]
possible_2018_losers_df = PAC_merge_df.loc[(PAC_merge_df['candidate_last_name'] != "") & (PAC_merge_df['Race'] == "")]
untraced_disbursements_df = PAC_merge_df.loc[PAC_merge_df['Race'] == ""]

PAC_race_df = winners_2018_df.groupby(['party','Race']).agg('size').unstack()
PAC_gender_df = winners_2018_df.groupby(['party','gender']).agg('size').unstack()
PAC_money_race_df = winners_2018_df.groupby(['party', 'Race'])['disbursement_amount'].agg('sum').unstack()
PAC_money_gender_df = winners_2018_df.groupby(['party', 'gender'])['disbursement_amount'].agg('sum').unstack()

#remove NaN
PAC_race_df = PAC_race_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
PAC_gender_df = PAC_gender_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
PAC_money_race_df = PAC_money_race_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
PAC_money_gender_df = PAC_money_gender_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))

#add total column
PAC_race_df['Total'] = race_totals(PAC_race_df)
PAC_gender_df['Total'] = gender_totals(PAC_gender_df)
PAC_money_race_df['Total'] = race_totals(PAC_money_race_df)
PAC_money_gender_df['Total'] = gender_totals(PAC_money_gender_df)
#PAC_race_df['Total'] = PAC_race_df['Asian'] + PAC_race_df['Black'] + PAC_race_df['Hispanic'] + PAC_race_df['Other Races'] + PAC_race_df['White']
#PAC_gender_df['Total'] = PAC_gender_df['Male'] + PAC_gender_df['Female']
#PAC_money_race_df['Total'] = PAC_money_race_df['Asian'] + PAC_money_race_df['Black'] + PAC_money_race_df['Hispanic'] + PAC_money_race_df['Other Races'] + PAC_money_race_df['White']
#PAC_money_gender_df['Total'] = PAC_money_gender_df['Male'] + PAC_money_gender_df['Female']

#concatenate race dataframes
PAC_race_df = pd.concat([PAC_race_df, us_race_df])
#concatenate gender dataframes
PAC_gender_df = pd.concat([PAC_gender_df, us_gender_df])

#convert floats to ints
PAC_race_df = PAC_race_df.convert_dtypes()
PAC_gender_df = PAC_gender_df.convert_dtypes()
PAC_money_race_df = PAC_money_race_df.convert_dtypes()
PAC_money_gender_df = PAC_money_gender_df.convert_dtypes()

#remove NaN
PAC_race_df = PAC_race_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
PAC_gender_df = PAC_gender_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
PAC_money_race_df = PAC_money_race_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))
PAC_money_gender_df = PAC_money_gender_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna(''))

#change index values for clearer presentation
change_index_values(PAC_race_df, "Democrat", "Democratic Recipiants")
change_index_values(PAC_race_df, "Republican", "Republican Recipiants")
change_index_values(PAC_gender_df, "Democrat", "Democratic Recipiants")
change_index_values(PAC_gender_df, "Republican", "Republican Recipiants")
change_index_values(PAC_money_race_df, "Democrat", "Democratic Recipiants")
change_index_values(PAC_money_race_df, "Republican", "Republican Recipiants")
change_index_values(PAC_money_gender_df, "Democrat", "Democratic Recipiants")
change_index_values(PAC_money_gender_df, "Republican", "Republican Recipiants")

#rearrange columns
race_cols = races
race_cols.append('Total')
gender_cols = genders
gender_cols.append('Total')
PAC_race_df = PAC_race_df[race_cols]
PAC_gender_df = PAC_gender_df[gender_cols]


# ## PAC disbursements based on inclusion and diversity. 
# ### In this analysis we take a closer look at the diversity of the United States overall and the diversity of the current US Congress by party.  We will analyze diversity based on gender and race and compare the parties in general and then dive into the diversity of the specific candidates that received money from the selected PAC.  All of this data is publicly available.

# In[10]:


display_gender_charts(gender_df.iloc[2:3], "US Population by Gender")


# In[11]:


display_gender_charts(gender_df.iloc[0:2], "Current Congress by Gender")


# In[12]:


display_gender_charts(PAC_gender_df.iloc[0:2], "Percent of PAC Disbursements to Members of the 116th Congress by Gender")


# In[13]:


display_gender_charts(PAC_gender_df.iloc[0:2].sum().to_frame(name='recipients').transpose(), "PAC Disbursement to candidates by Gender")


# ## PAC disbursements based on racial diversity

# In[14]:


display_race_charts(race_df.iloc[2:3], "US Population by Race")


# In[15]:


display_race_charts(race_df.iloc[0:2], "Current Congress by Race")


# In[16]:


display_race_charts(PAC_race_df.iloc[0:2], "Percent of PAC Disbursements to Members of the 116th Congress by Race")


# In[17]:


display_race_charts(PAC_race_df.iloc[0:2].sum().to_frame(name='recipients').transpose(), "PAC Disbursement to candidates by Race")


# In[18]:


display_money_charts(PAC_money_race_df, "Amount of PAC Disbursements to Members of the 116th Congress by Race")


# In[19]:


df = PAC_merge_df.groupby('Race')['disbursement_amount'].sum().sort_values()
as_list = df.index.tolist()
idx = as_list.index('')
as_list[idx] = 'candidates not elected or other PACs or RNCC/DCCC'
df.index = as_list


# In[20]:


display_bar(df)


# ## Sources
# - US demographics data (race) - https://en.wikipedia.org/wiki/Demographics_of_the_United_States#Race_and_ethnicity
# - US demographics data (gender) - https://www.census.gov/quickfacts/fact/table/US/LFE046218
# - 116th congress demographic data - https://adamisacson.com/116th-congress-spreadsheet/
# - FEC site - https://www.fec.gov
# - FEC 2017-2018 Google PAC disbursements - https://www.fec.gov/data/disbursements/?committee_id=C00428623&two_year_transaction_period=2018&data_type=processed

# In[ ]:




