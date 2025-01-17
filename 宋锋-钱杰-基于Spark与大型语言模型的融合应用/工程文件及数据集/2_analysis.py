from pyspark import SparkConf,SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import year
from pyspark.sql.functions import corr
from pyspark.sql.functions import month
from pyspark.sql.functions import to_date
from pyspark.sql.functions import when
from pyspark.sql.functions import col, to_date
from pyspark.sql.functions import sum
from pyspark.sql.functions import avg
from pyspark.sql.functions import stddev


spark = SparkSession.builder.getOrCreate()
df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('dataset/stock_everyday_data.csv')
df.createOrReplaceTempView("stock_data")
df.show(3)

# 将字符串日期转换为日期类型
df = df.withColumn("trade_date", to_date(col("trade_date").cast("string"), "yyyyMMdd"))


########1: 计算每年茅台股票的平均交易量########
# 将日期字段转换为年份
df = df.withColumn("year", year("trade_date"))
# 计算每年的平均交易量
yearly_avg_volume = df.groupBy("year").agg({"vol": "avg"}).orderBy("year")
# 显示结果
print("每年的平均交易量：")
yearly_avg_volume.show(100)
#保存在文件中
yearly_avg_volume.toPandas().to_csv("result/yearly_avg_volume.csv", index=True)



#########2: 分析股票收盘价的变化趋势########
# 计算收盘价的平均值和标准差
closing_stats = df.select(avg("close").alias("avg_close"), stddev("close").alias("stddev_close"))
# 显示结果
print("平均收盘价和标准差：")
closing_stats.show()



########3: 分析涨跌幅与各数据的的相关性########
# 计算涨跌幅与成交量的相关性
chg_volume_corr = df.select(corr("pct_chg", "vol").alias("chg_volume_corr"))
# 计算涨跌幅和peTTM的相关性
chg_pe_corr = df.select(corr("pct_chg", "peTTM").alias("chg_pe_corr"))
# 计算涨跌幅和pbMRQ的相关性
chg_pb_corr = df.select(corr("pct_chg", "pbMRQ").alias("chg_pb_corr"))
# 计算涨跌幅和psTTM的相关性
chg_ps_corr = df.select(corr("pct_chg", "psTTM").alias("chg_ps_corr"))
# 计算涨跌幅和pcfNcfTTM的相关性
chg_pcf_corr = df.select(corr("pct_chg", "pcfNcfTTM").alias("chg_pcf_corr"))
#合并上述结果在一张表中
correlation = chg_volume_corr.crossJoin(chg_pe_corr).crossJoin(chg_pb_corr).crossJoin(chg_ps_corr).crossJoin(chg_pcf_corr)
#显示结果
print("涨跌幅与各数据的相关性：")
correlation.show()
#保存在文件
correlation.toPandas().to_csv("result/correlation.csv", index=True)



########4: 计算各个月份股票的涨停和跌停次数########
# 将日期字段转换为月份
df = df.withColumn("month", month("trade_date"))
df = df.withColumn("pct_chg", df["pct_chg"].cast("float"))
df.show(3)
# 计算每月涨停和跌停的次数
monthly_limit_up_down = df.groupBy("month").agg(sum(when((df["pct_chg"] >= 10) | (df["pct_chg"] <= -10), 1).otherwise(0)).alias("limit_up_down_count"))
# 显示结果
print("各个月份涨停和跌停的次数：")
monthly_limit_up_down.orderBy("month").show()
#保存在文件中
monthly_limit_up_down.orderBy("month").toPandas().to_csv("result/monthly_limit_up_down.csv", index=True)


########5: 分析不同市场环境下的股票数据#######
#将涨跌幅字段转换为数值型
df = df.withColumn("pct_chg", df["pct_chg"].cast("float"))
# 定义市场环境规则并分类数据
df = df.withColumn("market_condition",
                       when(df["pct_chg"] > 2, "牛市")
                       .when(df["pct_chg"] < -2, "熊市")
                       .otherwise("波动市"))
# 统计不同市场环境下的股票数据
market_summary = df.groupBy("market_condition")\
                    .agg({"trade_date": "count", "vol": "sum","peTTM":"sum"})\
                    .withColumnRenamed("count(trade_date)", "num_of_days")\
                    .withColumnRenamed("sum(vol)", "total_volume")\
                    .withColumnRenamed("sum(peTTM)", "total_peTTM")
# 计算每个市场环境下的平均交易量
market_summary = market_summary.withColumn("avg_volume", col("total_volume") / col("num_of_days"))
# 计算平均peTTM
market_summary = market_summary.withColumn("avg_peTTM", col("total_peTTM") / col("num_of_days"))
# 显示结果
print("不同市场环境下的股票数据：")
market_summary.show()
#保存在文件中
market_summary.toPandas().to_csv("result/market_summary.csv", index=True)



########6: 分析近20年的最高股价和最低股价#######
sql_result = spark.sql("""SELECT trade_date, CAST(high AS DECIMAL) AS high_numeric
FROM stock_data
ORDER BY high_numeric DESC
LIMIT 1;""")
print("20年来贵州茅台最高股价:")
sql_result.show()

sql_result = spark.sql("""SELECT trade_date, CAST(low AS DECIMAL) AS low_numeric
FROM stock_data
ORDER BY low_numeric ASC
LIMIT 1;""")
print("20年来贵州茅台最低股价:")
sql_result.show()



########7: 计算上周的平均涨幅#######
sql_result = spark.sql("""SELECT AVG(pct_chg) AS avg_chg
FROM stock_data
WHERE trade_date BETWEEN '20240505' AND '20240510';
""")
print("上周平均涨幅:")
sql_result.show()



########8: 统计每年上涨和下跌天数#######
# 计算每月涨停和跌停的次数
year_up_days = df.groupBy("year").agg(sum(when((df["pct_chg"] >= 0), 1).otherwise(0)).alias("up_count"))
year_down_days = df.groupBy("year").agg(sum(when((df["pct_chg"] < 0), 1).otherwise(0)).alias("down_count"))
#合并在同一张表并按年份排序
year_up_down_days = year_up_days.join(year_down_days, "year").orderBy("year")
#添加列：上涨天数占总天数的比值
year_up_down_days = year_up_down_days.withColumn("up_ratio", col("up_count") / (col("up_count") + col("down_count")))
#添加列：下跌天数占总天数的比值
year_up_down_days = year_up_down_days.withColumn("down_ratio", col("down_count") / (col("up_count") + col("down_count")))
print("每年上涨和下跌天数：")
year_up_down_days.show(100)
#保存在文件中
year_up_down_days.toPandas().to_csv("result/year_up_down_days.csv", index=True)
