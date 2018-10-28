from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_page(x):
    """
    2NE1_zh.wikipedia.org_all-access_spider  country site
    :param x:
    :return:
    """
    x = x.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]


def nan_fill_forward(x):
    for i in range(x.shape[0]):
        fill_val = None
        for j in range(x.shape[1] - 3, x.shape[1]):
            if np.isnan(x[i, j]) and fill_val is not None:
                x[i, j] = fill_val
            else:
                fill_val = x[i, j]
    return x


def create_final_csv():
    # df1 = pd.read_csv('data/raw/train_1.csv', encoding='utf-8')
    # 803天 2015-07-01-2017-09-10
    df2 = pd.read_csv('data/raw/train_2.csv', encoding='utf-8')
    # 28天 2017-08-15-2017-09-11
    scraped = pd.read_csv('data/raw/2017-08-15_2017-09-11.csv', encoding='utf-8')
    # Update last two days by scraped data
    df2['2017-09-10'] = scraped['2017-09-10']
    df2['2017-09-11'] = scraped['2017-09-11']
    return df2


df = create_final_csv().iloc[0:2048]
# 日期列
date_cols = [i for i in df.columns if i != 'Page']
# url分成四个部分
df['name'], df['project'], df['access'], df['agent'] = zip(*df['Page'].apply(parse_page))

# 需要onehot
le = LabelEncoder()
df['project'] = le.fit_transform(df['project'])
df['access'] = le.fit_transform(df['access'])
df['agent'] = le.fit_transform(df['agent'])
# page_id 保留序号
df['page_id'] = le.fit_transform(df['Page'])

if not os.path.isdir('data/processed'):
    os.makedirs('data/processed')

df[['page_id', 'Page']].to_csv('data/processed/page_ids.csv', encoding='utf-8', index=False)

data = df[date_cols].values
# 保留销量矩阵 nan转成0
np.save('data/processed/data.npy', np.nan_to_num(data))
# 空值 true false 矩阵 -->0 1
np.save('data/processed/is_nan.npy', np.isnan(data).astype(int))
#
np.save('data/processed/project.npy', df['project'].values)
np.save('data/processed/access.npy', df['access'].values)
np.save('data/processed/agent.npy', df['agent'].values)
np.save('data/processed/page_id.npy', df['page_id'].values)

# 填充最后三天的值 向后填充
test_data = nan_fill_forward(df[date_cols].values)
np.save('data/processed/test_data.npy', np.nan_to_num(test_data))
np.save('data/processed/test_is_nan.npy', np.isnan(test_data).astype(int))
