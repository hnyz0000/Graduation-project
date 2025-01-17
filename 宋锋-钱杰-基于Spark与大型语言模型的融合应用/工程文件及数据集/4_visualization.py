from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# 读取CSV文件，并设置第0列作为索引
yearly_avg_volume = pd.read_csv("result/yearly_avg_volume.csv", index_col=0)

# 创建图表，并设置图表大小
plt.figure(figsize=(12, 6))
# 绘制柱状图，X轴为年份，Y轴为平均交易量，颜色为蓝色
plt.bar(yearly_avg_volume['year'], yearly_avg_volume['avg(vol)'], color='b', label='Average Volume')

# 设置图表标题和X、Y轴标签
plt.title('Yearly Average Stock Volume')
plt.xlabel('Year')
plt.ylabel('Average Volume')
# 保存图表为PNG文件
plt.savefig("result/yearly_avg_volume.png")


######## 可视化收盘价与其他数据的相关性 ########
# 读取相关性数据文件
correlation = pd.read_csv("result/correlation.csv", index_col=0)

# 创建图表，并设置图表大小
plt.figure(figsize=(12, 6))
# 绘制柱状图，X轴为不同数据列，Y轴为相关性值，颜色为蓝色
plt.bar(correlation.columns, correlation.iloc[0], color='b', label='Correlation')

# 设置图表标题和X、Y轴标签
plt.title('Correlation between Closing Price and Other Data')
plt.xlabel('Data')
plt.ylabel('Correlation')
# 保存图表为PNG文件
plt.savefig("result/correlation.png")


######## 可视化每年股票上涨和下跌天数的比例 ########
# 读取每年股票上涨和下跌天数的比例数据
year_up_down_days = pd.read_csv("result/year_up_down_days.csv", index_col=0)

# 将下跌天数比例变为负值
year_up_down_days['down_ratio'] = -year_up_down_days['down_ratio']
# 创建图表，并设置图表大小
plt.figure(figsize=(12, 6))
# 绘制两组柱状图，绿色代表上涨天数比例，红色代表下跌天数比例
plt.bar(year_up_down_days['year'], year_up_down_days['up_ratio'], color='g', label='Up Days Ratio')
plt.bar(year_up_down_days['year'], year_up_down_days['down_ratio'], color='r', label='Down Days Ratio')

# 设置图表标题和X、Y轴标签
plt.title('Yearly Stock Up and Down Days Ratio')
plt.xlabel('Year')
plt.ylabel('Ratio')
# 保存图表为PNG文件
plt.savefig("result/year_up_down_days_ratio.png")


######## 可视化每个月涨跌停次数 ########
# 读取每个月涨跌停次数数据
month_limit_up_down = pd.read_csv("result/monthly_limit_up_down.csv", index_col=0)

# 创建图表，并设置图表大小
plt.figure(figsize=(12, 12))
# 删除值为0的月份数据
month_limit_up_down = month_limit_up_down[month_limit_up_down['limit_up_down_count'] != 0]
# 绘制饼状图，显示不同月份的涨跌停次数比例
plt.pie(month_limit_up_down['limit_up_down_count'], labels=month_limit_up_down.index, autopct='%1.1f%%', startangle=90)
# 设置图表标题
plt.title('Monthly Limit Up and Down Days Count')
# 保存图表为PNG文件
plt.savefig("result/monthly_limit_up_down.png")


######## 可视化不同市场环境下的股票数据 ########
# 读取市场环境数据
market_env = pd.read_csv("result/market_summary.csv", index_col=0)

# 创建图表和坐标轴
fig, ax = plt.subplots(figsize=(12, 6))
# 定义指标名称
indicators = ['total_peTTM', 'total_volume', 'num_of_days', 'avg_volume', 'avg_peTTM']
# 设置缩放因子，用于缩放部分指标
scale_factors = {'total_peTTM': 1e-5, 'total_volume': 1e-8, 'num_of_days': 1e-3, 'avg_volume': 1e-5, 'avg_peTTM': 1e-1}
# 准备数据
n_groups = len(indicators)
index = np.arange(n_groups)  # 指标数量
bar_width = 0.25  # 每个柱状图的宽度
opacity = 0.8  # 透明度

# 循环绘制每个市场状态
condition_name = ['sideway','bull','bear']
for i, condition in enumerate(market_env.index):
    # 缩放数据
    values = market_env.loc[condition, indicators] * pd.Series(scale_factors)
    # 绘制每个市场状态下的柱状图
    plt.bar(index + i * bar_width, values, bar_width, alpha=opacity, label=f'Market Condition: {condition_name[i]}')

# 设置图表标题和X、Y轴标签
plt.title('Comparison of Market Conditions Across Different Indicators')
plt.xlabel('Indicators')
plt.ylabel('Scaled Values')
# 设置X轴标签的位置
plt.xticks(index + bar_width, indicators)

# 添加图例
plt.legend()

# 调整布局并显示图表
plt.tight_layout()
# 保存图表为PNG文件
plt.savefig("result/market_summary.png")


# 创建Spark会话（注释掉了Spark会话创建代码）
# spark = SparkSession.builder.getOrCreate()

# 读取股票每日数据的CSV文件
df = pd.read_csv(r"C:\Users\hnyz123\Desktop\PythonProject\宋锋-钱杰-基于Spark与大型语言模型的融合应用\工程文件及数据集\dataset\stock_everyday_data.csv")
# df = pd.read_csv("宋锋-钱杰-基于Spark与大型语言模型的融合应用/工程文件及数据集/dataset/stock_everyday_data.csv")

# 将'trade_date'列转换为日期类型
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

# 提取最后300条数据
pandas_df = df[-300:]

# 获取最大和最小的收盘价
max_close = pandas_df['close'].max()
min_close = pandas_df['close'].min()

# 将'trade_date'列转换为列表
trade_date_list = pandas_df['trade_date'].tolist()
# 将'close'列转换为列表
close_list = pandas_df['close'].tolist()

# 创建图表并设置图表大小
plt.figure(figsize=(25, 12))
# 绘制收盘价的折线图
plt.plot(trade_date_list, close_list, marker='o', linestyle='-', color='royalblue', markersize=1, linewidth=2, label='Close Price')
# 设置图表标题
plt.title('Price Trend', fontsize=30)
plt.xlabel('Trade Date', fontsize=20)
plt.ylabel('Closing Price', fontsize=20)
# 设置X轴标签旋转角度
plt.xticks(rotation=45)
# 设置网格线
plt.grid(True, linestyle='--', linewidth=0.5)
# 设置坐标轴标签的字体大小
plt.tick_params(axis='both', labelsize=14)
# 设置Y轴范围
plt.ylim(min_close * 0.9, max_close * 1.1)
# 格式化X轴日期显示
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# 自动调整X轴标签的显示间隔
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
# 调整布局以防止标签重叠
plt.tight_layout()

# 保存图表为PNG文件
plt.savefig('stock_price_trend_enhanced.png')
