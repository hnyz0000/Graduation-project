# 主要任务是从 Hugging Face Hub 下载一个指定的模型，并将其缓存到本地以供后续使用。
# 首先连接到 Hugging Face 模型库（huggingface.co），并从中下载指定的预训练模型。
# 这些模型可以用于自然语言处理、计算机视觉等任务。
# 如果使用的是类似 transformers、datasets 等库，会自动从 Hugging Face Hub 拉取模型。
# 避免警告
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 导入必要的库
# 用于与操作系统交互，如设置环境变量

# 设置 Hugging Face 的端点为镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置 CUDA 版本（在运行深度学习模型时，需要选择合适的 CUDA 版本）
os.environ['CUDA_VERSION'] = "118"

# 设置使用的 GPU 设备编号，这里选择 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 导入 PyTorch 的功能库
import torch.fx  # 用于跟踪和转换 PyTorch 模型的功能
import torch  # 用于执行深度学习模型的计算

# 导入 Hugging Face Transformers 库中的模型和 tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入用于 SQL 查询格式化的 sqlparse 库
import sqlparse

# 导入 PySpark 库，用于创建 Spark 会话，执行分布式 SQL 查询
from pyspark.sql import SparkSession

# 创建 Spark 会话，Spark 会话是与 Spark 集群交互的入口
spark = SparkSession.builder.getOrCreate()

# 加载数据集（指定 CSV 文件路径）
df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("C:/Users/hnyz123/Desktop/PythonProject/宋锋-钱杰-基于Spark与大型语言模型的融合应用/工程文件及数据集/dataset/stock_everyday_data.csv")

# 将除了 trade_date 列以外的所有列转换为浮动数值类型，以便进行数值计算
for col in df.columns:
    if col != "trade_date":  # 排除 trade_date 列，因为其是日期类型
        df = df.withColumn(col, df[col].cast("float"))

# 将 DataFrame 注册为临时 SQL 表格，方便使用 Spark SQL 查询
df.createOrReplaceTempView("a_stock_everyday_data")

# 加载 Text2SQL 模型，这个模型可以生成 SQL 查询
model_name = "defog/sqlcoder-7b-2"  # 指定模型的名称
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 加载用于文本处理的 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,  # 允许加载远程模型代码
    torch_dtype=torch.float16,  # 使用半精度 float16 来减少显存占用
    device_map="auto",  # 自动选择设备（GPU 或 CPU）
    use_cache=True,  # 启用缓存，提高推理效率
)

# 定义用于生成 SQL 查询的模板
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:

CREATE TABLE a_stock_everyday_data (
  trade_date DATE, -- Trading date交易日期，以纯数字形式给出，例如20240515
  open DECIMAL(10,2), -- Opening price开盘价
  high DECIMAL(10,2), -- Highest price当日最高价
  low DECIMAL(10,2), -- Lowest price当日最低价
  close DECIMAL(10,2), -- Closing price收盘价
  pre_close DECIMAL(10,2), -- Previous closing price前一天的收盘价
  change DECIMAL(10,2), -- Price change股价变化量
  pct_chg DECIMAL(10,2), -- Price change percentage涨幅
  vol BIGINT, -- Trading volume成交额
  amount DECIMAL(15,2), -- Trading amount成交量
);

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
"""  # 这部分模板提供了表结构和问题上下文，以便模型根据用户的问题生成相应的 SQL 查询

# 定义生成 SQL 查询的函数
def generate_query(question):
    # 将用户输入的问题插入模板中
    updated_prompt = prompt.format(question=question)

    # 使用 tokenizer 将输入的文本转换为模型可以理解的格式
    inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")

    # 生成 SQL 查询
    generated_ids = model.generate(
        **inputs,  # 将处理后的输入传入模型
        num_return_sequences=1,  # 只生成一个查询
        eos_token_id=tokenizer.eos_token_id,  # 生成的结束标记
        pad_token_id=tokenizer.eos_token_id,  # 填充标记
        max_new_tokens=200,  # 限制生成最多 400 个 token
        do_sample=False,  # 不使用采样方法，确保生成结果一致
        num_beams=1,  # 使用束搜索算法来生成最优答案
    )

    # 解码生成的 token ID 为字符串
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # 清理显存，确保在每次推理后释放 GPU 显存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 提取 SQL 查询并返回
    return outputs[0].split("```sql")[1].split(";")[0]  # 提取 SQL 查询部分


# 启动一个交互式循环，允许用户输入问题
while (True):
    # 提示用户输入问题
    question = input("请输入一个问题：")

    # 如果用户输入“exit”，则退出循环
    if question == "exit":
        break

    # 调用函数生成 SQL 查询
    generated_sql = generate_query(question)

    # 打印生成的 SQL 查询，使用 sqlparse 格式化输出
    print("->LLM生成的SQL语句为：", end='')
    print(sqlparse.format(generated_sql, reindent=True))  # 格式化 SQL 查询以便于阅读

    # 执行生成的 SQL 查询并显示结果
    sql_result = spark.sql(generated_sql)  # 使用 Spark 执行 SQL 查询
    print("->查询结果为：")
    sql_result.show()  # 显示查询结果
