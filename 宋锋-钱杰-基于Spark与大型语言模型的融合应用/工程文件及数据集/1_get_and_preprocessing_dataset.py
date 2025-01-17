import tushare as ts
import pandas as pd
import numpy as np
from warnings import simplefilter
import baostock as bs

simplefilter(action="ignore", category=FutureWarning)

# 设置tushare API token
ts.set_token('6f4256432f33166f825436227e51f0cfd84eeac2e9da2bf808554820')

pro = ts.pro_api()
lg = bs.login()

# 定义获取数据的日期范围和指定股票代码
start_date = '20040101'
end_date = '20241230'
ts_code = "600519.SH" # 贵州茅台对应股票代码

hist_data = pro.daily(ts_code=ts_code, start_date=start_date,
                          end_date=end_date)

hist_data = hist_data[::-1]  # 倒序，日期从小到大

# 从另外一个数据来源获取滚动市盈率和滚动市销率
k_data = bs.query_history_k_data_plus(ts_code, 'code,date,peTTM,pbMRQ,psTTM,pcfNcfTTM',
                                        start_date=start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:],
                                        end_date=end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:])
k_data_df = k_data.get_data()

# 使用pandas进行格式化统一不同数据来源的日期格式以便下一步合并DataFrame
k_data_df['date'] = pd.to_datetime(k_data_df['date'])
k_data_df['trade_date'] = k_data_df['date'].apply(lambda x: x.strftime('%Y%m%d'))
k_data_df = k_data_df.drop(['date', 'code'], axis=1)

# 使用pandas合并两个DataFrame
# 将每日K线数据和每日滚动市盈率、滚动市销率合并到一个DataFrame中
hist_data = pd.merge(hist_data, k_data_df, on='trade_date')

# 部分字段存在缺失现象，根据股票数据的特点，使用前值填充的方式填充缺失值
hist_data['peTTM'].fillna(method='ffill', inplace=True)
hist_data['pbMRQ'].fillna(method='ffill', inplace=True)
hist_data['psTTM'].fillna(method='ffill', inplace=True)
hist_data['pcfNcfTTM'].fillna(method='ffill', inplace=True)

# 部分股票数据存在异常0值（这些值不应该为0），仍使用前值填充的方式填充0值
hist_data['peTTM'].replace(0, method='ffill', inplace=True)
hist_data['pbMRQ'].replace(0, method='ffill', inplace=True)
hist_data['psTTM'].replace(0, method='ffill', inplace=True)
hist_data['pcfNcfTTM'].replace(0, method='ffill', inplace=True)

# 将股票数据存储为CSV文件
filename = 'dataset/stock_everyday_data.csv'

hist_data.to_csv(filename, index=False)
