========
行情API
========

接口概要
========

开放接口基于PC客户端获取数据，提供给用户使用。

开放接口分为\ **低频接口**\ 、\ **订阅接口**\ 和\ **高频接口**\ ，以及\ **回调处理基类**\ ：

**低频接口**\ 主要用来获取股票市场静态和全局的数据，让用户得到股票的基本信息，不允许高频调用。

如果要实时获取数据，则需要调用高频接口。

**订阅接口**\ 是用来管理高频接口使用额度，包括订阅、退订和查询额度。

*订阅*\ ：在使用高频接口前， 需要订阅想要的数据。订阅成功后，则可以使用高频接口获取；订阅各类数据有额度限制：

**用户额度 >= K线订阅数 \* K线权重 + 逐笔订阅数 \* 逐笔权重 + 报价订阅数
\* 报价权重 + 摆盘订阅数 \* 摆盘权重**

**订阅使用的额度不能超过用户额度，用户额度也就是订阅的上限为500个订阅单位**

+------------+----------------------------+
| 订阅数据   | 额度权重（所占订阅单位）   |
+============+============================+
| K线        | 2                          |
+------------+----------------------------+
| 逐笔       | 5（牛熊证为1）             |
+------------+----------------------------+
| 报价       | 1                          |
+------------+----------------------------+
| 摆盘       | 5（牛熊证为1）             |
+------------+----------------------------+
| 分时       | 2                          |
+------------+----------------------------+
| 经纪队列   | 5（牛熊证为1）             |
+------------+----------------------------+

*查询额度*\ ：用来查询现在各项额度占用情况。用户可以看到每一种类型数据都有订阅了哪些股票；然后利用退订操作来去掉不需要的股票数据。

*退订*\ ：用户可以退订指定股票和指定数据类型，空出额度。但是退订的时间限制为1分钟，即订阅某支股票某个订阅位1分钟之后才能退订。

**如果数据不订阅，直接调用高频接口则会返回失败。**
订阅时可选择推送选项。推送开启后，程序就会持续收到客户端推送过来的行情数据。用户可以通过继承\ **回调处理基类**\ ，并实现用户自己的子类来使用数据推送功能。

**高频接口**\ 可以获取实时数据，可以针对小范围内的股票频繁调用；比如需要跟踪某个股票的逐笔和摆盘变化等；在调用之前需要将频繁获取的数据订阅注册。

**回调处理基类**\ 用于实现数据推送功能，用户在此基类上实现子类并实例化后，当客户端不断推送数据时，程序就会调用对应的对象处理。

接口列表
=========

上下文控制
~~~~~~~~~~

.. code:: python

    start()              # 开启异步数据接收

    stop()               # 停止异步数据接收

    set_handler(handler) # 设置用于异步处理数据的回调对象

低频数据接口
~~~~~~~~~~~~

.. code:: python

    get_trading_days(market, start_date=None, end_date=None)  # 获取交易日
    get_stock_basicinfo(market, stock_type='STOCK')           # 获取股票信息
    get_history_kline(code, start=None, end=None, ktype='K_DAY', autype='qfq')  # 获取历史K线
    get_autype_list(code_list)      # 获取复权因子
    get_market_snapshot(code_list)  # 获取市场快照
    get_plate_list(market, plate_class)        #获取板块集合下的子板块列表
    get_plate_stock(market, stock_code)        #获取板块下的股票列表

订阅接口
~~~~~~~~

.. code:: python

    subscribe(stock_code, data_type, push=False) # 订阅
    unsubscribe(stock_code, data_type)           # 退订
    query_subscription()                         # 查询订阅

高频数据接口
~~~~~~~~~~~~

.. code:: python

    get_stock_quote(code_list) #  获取报价
    get_rt_ticker(code, num)   # 获取逐笔
    get_cur_kline(code, num, ktype=' K_DAY', autype='qfq') # 获取当前K线
    get_order_book(code)       # 获取摆盘
    get_rt_data                #获取分时数据
    get_broker_queue           #获取经纪队列

回调处理基类
~~~~~~~~~~~~

.. code:: python

    StockQuoteHandlerBase # 报价处理基类

    OrderBookHandlerBase  # 摆盘处理基类

    CurKlineHandlerBase   # 实时K线处理基类

    TickerHandlerBase     # 逐笔处理基类

    RTDataHandlerBase     # 分时数据处理基类

    BrokerHandlerBase     # 经纪队列处理基类

参数类型定义
============

市场标识market
~~~~~~~~~~~~~~
（字符串类型）

+------------+----------------+
| 股票市场   | 标识           |
+============+================+
| 港股       | "HK"           |
+------------+----------------+
| 美股       | "US"           |
+------------+----------------+
| 沪股       | "SH"           |
+------------+----------------+
| 深股       | "SZ"           |
+------------+----------------+
| 香港期货   | "HK\_FUTURE"   |
+------------+----------------+

证券类型stock\_type
~~~~~~~~~~~~~~~~~~~
（字符串类型）

+------------+-------------+
| 股票类型   | 标识        |
+============+=============+
| 正股       | "STOCK"     |
+------------+-------------+
| 指数       | "IDX"       |
+------------+-------------+
| ETF基金    | "ETF"       |
+------------+-------------+
| 涡轮牛熊   | "WARRANT"   |
+------------+-------------+
| 债券       | "BOND"      |
+------------+-------------+

复权类型autype
~~~~~~~~~~~~~~
（字符串类型）

+------------+---------+
| 复权类型   | 标识    |
+============+=========+
| 前复权     | "qfq"   |
+------------+---------+
| 后复权     | "hfq"   |
+------------+---------+
| 不复权     | None    |
+------------+---------+

K线类型
~~~~~~~~
（字符串类型）

+-----------+-------------+
| K线类型   | 标识        |
+===========+=============+
| 1分K      | "K\_1M"     |
+-----------+-------------+
| 5分K      | "K\_5M"     |
+-----------+-------------+
| 15分K     | "K\_15M"    |
+-----------+-------------+
| 30分K     | "K\_30M"    |
+-----------+-------------+
| 60分K     | "K\_60M"    |
+-----------+-------------+
| 日K       | "K\_DAY"    |
+-----------+-------------+
| 周K       | "K\_WEEK"   |
+-----------+-------------+
| 月K       | "K\_MON"    |
+-----------+-------------+

订阅数据类型
~~~~~~~~~~~~
（字符串类型）

+------------+-----------------+
| 订阅类型   | 标识            |
+============+=================+
| 逐笔       | "TICKER"        |
+------------+-----------------+
| 报价       | "QUOTE"         |
+------------+-----------------+
| 摆盘       | "ORDER\_BOOK"   |
+------------+-----------------+
| 1分K       | "K\_1M"         |
+------------+-----------------+
| 5分K       | "K\_5M"         |
+------------+-----------------+
| 15分K      | "K\_15M"        |
+------------+-----------------+
| 30分K      | "K\_30M"        |
+------------+-----------------+
| 60分K      | "K\_60M"        |
+------------+-----------------+
| 日K        | "K\_DAY"        |
+------------+-----------------+
| 周K        | "K\_WEEK"       |
+------------+-----------------+
| 月K        | "K\_MON"        |
+------------+-----------------+
| 分时       | "RT\_DATA"      |
+------------+-----------------+
| 经纪队列   | "BROKER"        |
+------------+-----------------+

板块分类类型
~~~~~~~~~~~~
（字符串类型）

+------------+--------------+
| 板块分类   | 标识         |
+============+==============+
| 所有板块   | "ALL"        |
+------------+--------------+
| 行业分类   | "INDUSTRY"   |
+------------+--------------+
| 地域分类   | "REGION"     |
+------------+--------------+
| 概念分类   | "CONCEPT"    |
+------------+--------------+

K线字段类型
~~~~~~~~~~~~

+-------------------------+------------+
| 字段标识                | 说明       |
+=========================+============+
| KL\_FIELD.ALL           | 所有字段   |
+-------------------------+------------+
| KL\_FIELD.DATE_TIME     | K线时间    |
+-------------------------+------------+
| KL\_FIELD.OPEN          | 开盘价     |
+-------------------------+------------+
| KL\_FIELD.CLOSE         | 收盘价     |
+-------------------------+------------+
| KL\_FIELD.HIGH          | 最高价     |
+-------------------------+------------+
| KL\_FIELD.LOW           | 最低价     |
+-------------------------+------------+
| KL\_FIELD.PE_RATIO      | 市盈率     |
+-------------------------+------------+
| KL\_FIELD.TRADE_VOL     | 成交量     |
+-------------------------+------------+
| KL\_FIELD.TRADE_VAL     | 成交额     |
+-------------------------+------------+
| KL\_FIELD.CHANGE_RATE   | 涨跌幅     |
+-------------------------+------------+

股票代码
~~~~~~~~
模式为：“ 市场+原始代码" 例如，“HK.00700”, “SZ.000001”,
“US.AAPL”

**注意，原始代码部分的字符串必须和客户端显示的完全匹配**，比如：
腾讯为“HK.00700”，而不能是“HK.700”

返回值
~~~~~~~

**对于用户来说接口会返回两个值**
ret\_code(调用执行返回状态，0为成功，其它为失败)和ret\_data：

+-------------+-----------------------------+
| ret\_code   | ret\_data                   |
+=============+=============================+
| 成功        | ret\_data为实际数据         |
+-------------+-----------------------------+
| 失败        | ret\_data为错误描述字符串   |
+-------------+-----------------------------+

错误码说明
##########

+----------+------------------------+
| 错误码   | 错误说明               |
+==========+========================+
| 0        | 无错误                 |
+----------+------------------------+
| 400      | 未知错误               |
+----------+------------------------+
| 401      | 版本不支持             |
+----------+------------------------+
| 402      | 股票未找到             |
+----------+------------------------+
| 403      | 协议号不支持           |
+----------+------------------------+
| 404      | 参数错误               |
+----------+------------------------+
| 405      | 频率错误（请求过快）   |
+----------+------------------------+
| 406      | 订阅达到上限           |
+----------+------------------------+
| 407      | 未订阅                 |
+----------+------------------------+
| 408      | 未满足反订阅时间限制   |
+----------+------------------------+
| 501      | 服务器忙               |
+----------+------------------------+
| 502      | 超时                   |
+----------+------------------------+
| 503      | 网络错误               |
+----------+------------------------+
| 504      | 操作不允许             |
+----------+------------------------+
| 505      | 未知订单               |
+----------+------------------------+

详细说明
=========

实例化上下文对象
~~~~~~~~~~~~~~~~~

.. code:: python

    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

**功能**\ ：创建上下文，建立网络连接

**参数**:

**host**\ ：网络连接地址

**sync\_port**\ ：网络连接端口，用于同步通信。

**async\_port**\ ：网络连接端口，用于异步通信，接收客户端的数据推送。

启动推送接收 start
~~~~~~~~~~~~~~~~~~~

.. code:: python

    quote_ctx.start()

**功能**\ ：启动推送接收线程，异步接收客户端推送的数据。

停止推送接收 stop
~~~~~~~~~~~~~~~~~~

.. code:: python

    quote_ctx.stop()

**功能**\ ：停止推送接收线程，不再接收客户端推送的数据。

设置异步回调处理对象 set\_handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    quote_ctx.set_handler(handler)

**功能**\ ：设置回调处理对象，用于接收线程在收到数据后调用。用户应该将自己实现的回调对象设置，以便实现事件驱动。

handler必须是以下几种类的子类对象：

+-----------------------------+--------------------+
| 类名                        | 说明               |
+=============================+====================+
| **StockQuoteHandlerBase**   | 报价处理基类       |
+-----------------------------+--------------------+
| **OrderBookHandlerBase**    | 摆盘处理基类       |
+-----------------------------+--------------------+
| **CurKlineHandlerBase**     | 实时K线处理基类    |
+-----------------------------+--------------------+
| **TickerHandlerBase**       | 逐笔处理基类       |
+-----------------------------+--------------------+
| **RTDataHandlerBase**       | 分时数据处理基类   |
+-----------------------------+--------------------+
| **BrokerHandlerBase**       | 经纪队列处理基类   |
+-----------------------------+--------------------+

获取交易日 get\_trading\_days
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_trading_days(market, start_date=None, end_date=None)

**功能**\ ：取指定市场，某个日期时间段的交易日列表

**参数**\ ：

**market**: 市场标识

| **start\_date**: 起始日期;
| 
  string类型，格式YYYY-MM-DD，仅指定到日级别即可，默认值None表示最近一年前的日期

**end\_date**: 结束日期;
string类型，格式YYYY-MM-DD，仅指定到日级别即可，取默认值None表示取当前日期

**返回**\ ：

ret\_code失败时，ret\_data返回为错误描述字符串；
客户端无数据时，ret\_code为成功，ret\_data返回None
正常情况下，ret\_data为日期列表（每个日期是string类型），如果指定时间段中无交易日，则ret\_data为空列表。

**失败情况**\ ：

1. 市场标识不合法

2. 起止日期输入不合法

3. 客户端内部或网络错误

获取股票信息 get\_stock\_basicinfo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_stock_basicinfo(market, stock_type='STOCK')

**功能**\ ：取符合市场和股票类型条件的股票简要信息

**参数**\ ：

**market**: 市场标识, string，例如，”HK”，”US”；具体见市场标识说明

**stock\_type**: 证券类型, string,
例如，”STOCK”，”ETF”；具体见证券类型说明

**证券类型** stock\_type，（字符串类型）：

+------------+-------------+
| 股票类型   | 标识        |
+============+=============+
| 正股       | "STOCK"     |
+------------+-------------+
| 指数       | "IDX"       |
+------------+-------------+
| ETF基金    | "ETF"       |
+------------+-------------+
| 涡轮牛熊   | "WARRANT"   |
+------------+-------------+
| 债券       | "BOND"      |
+------------+-------------+

**返回**\ ： ret\_code失败时，ret\_data返回为错误描述字符串；
客户端无符合条件数据时，ret\_code为成功，ret\_data返回None
正常情况下，ret\_data为一个dataframe，其中包括：

**code**\ ：股票代码；string，例如： ”HK.00700”，“US.AAPL”

**name**\ ：股票名称；string

**lot\_size**\ ：每手股数；int

**stock\_type**\ ：股票类型；string，例如： ”STOCK”，”ETF”

**stock\_child\_type**:
股票子类型；仅支持窝轮，其他为0，string，例如："BEAR"，"BULL"

**owner\_stock\_code**\ ：所属正股；仅支持窝轮，其他为0

**listing\_date**: 上市日期： str

**stockid**: 股票ID： str

**失败情况**\ ：

1. 市场或股票类型不合法

2. 客户端内部或网络错误

获取复权因子 get\_autype\_list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_autype_list(code_list)

**功能**\ ：获取复权因子数据

**参数**\ ：

**code\_list**: 股票代码列表，例如，HK.00700，US.AAPL

**返回**\ ：

ret\_code失败时，ret\_data返回为错误描述字符串；
客户端无符合条件数据时，ret\_code为成功，ret\_data返回None
正常情况下，ret\_data为一个dataframe，其中包括：

**code**\ ：股票代码，string，例如： ”HK.00700”，“US.AAPL”

**ex\_div\_date**\ ：除权除息日，string类型，格式YYYY-MM-DD

**split\_ratio**\ ：拆合股比例
double，例如，对于5股合1股为1/5，对于1股拆5股为5/1

**per\_cash\_div**\ ：每股派现；double

**per\_share\_div\_ratio**\ ：每股送股比例； double

**per\_share\_trans\_ratio**\ ：每股转增股比例； double

**allotment\_ratio**\ ： 每股配股比例；double

**allotment\_price**\ ：配股价；double

**stk\_spo\_ratio**\ ： 增发比例：double

**stk\_spo\_price** 增发价格：double

**forward\_adj\_factorA**\ ：前复权因子A；double

**forward\_adj\_factorB**\ ：前复权因子B；double

**backward\_adj\_factorA**\ ：后复权因子A；double

**backward\_adj\_factorB**\ ：后复权因子B；double

返回数据中不一定包含所有codelist中的代码，调用方自己需要检查，哪些股票代码是没有返回复权数据的，未返回复权数据的股票说明没有找到相关信息。

**复权价格 = 复权因子A \* 价格 + 复权因子B**

**失败情况**\ ：

1． Codelist中股票代码不合法

2． 客户端内部或网络错误

获取历史K线 get\_history\_kline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_history_kline(code, start=None, end=None, ktype='K_DAY', autype='qfq', fields=[KL_FIELD.ALL])

**功能**\ ： 获取指定股票K线历史数据

**参数**\ ：

**code**\ ：股票代码

**start** ：开始时间, string; YYYY-MM-DD；为空时取去当前时间;

**end** ： 结束时间, string; YYYY-MM-DD；为空时取当前时间;

**ktype** ：k线类型，默认为日K

**autype**:  复权类型，string；”qfq”-前复权，”hfq”-后复权，None-不复权，默认为”qfq”

**fields**: 单个或多个K线字段类型，指定需要返回的数据 KL_FIELD.ALL or [KL_FIELD.DATE_TIME, KL_FIELD.OPEN]，默认为KL_FIELD.ALL

开始结束时间按照闭区间查询，时间查询以k线时间time\_key作为比较标准。即满足
start<=Time\_key<=end条件的k线作为返回内容，k线时间time\_key的设定标准在返回值中说明

**返回**\ ：

| ret\_code失败时，ret\_data返回为错误描述字符串；
  客户端无符合条件数据时，ret\_code为成功，返回None

正常情况下返回K线数据为一个DataFrame包含:

**code**\ ： 股票代码；string

**time\_key**\ ： K线时间 string “YYYY-MM-DD HH:mm:ss”

**open**\ ： 开盘价；double

**high**\ ： 最高价；double

**close**\ ： 收盘价；double

**low**\ ： 最低价：double

**volume**\ ： 成交量；long

**turnover** ：成交额；double

**pe_ratio**：市盈率 ：double

**turnover_rate**:  换手率：double

**change_rate**:   涨跌幅：double

对于日K线，time\_key为当日时间具体到日，比如说2016-12-23日的日K，K线时间为”2016-12-23 00:00:00”

对于周K线，12月19日到12月25日的周K线，K线时间time\_key为” 2016-12-19 00:00:00”

对于月K线，12月的月K线时间time\_key为” 2016-12-01 00:00:00”，即为当月1日时间

对于分K线，time\_key为当日时间具体到分，例如，

+------------+-------------------------------------------------------------+
| 分K类型    | 覆盖时间举例                                                |
+============+=============================================================+
| 1分K       | 覆盖9:35:00到9:35:59的分K,time\_key为"2016-12-23 09:36:00"  |
+------------+-------------------------------------------------------------+
| 5分K       | 覆盖10:05:00到10:09:59的分K,time\_key为"2016-12-23 10:10:00"|
+------------+-------------------------------------------------------------+
| 15分K      | 覆盖10:00:00到10:14:59的分K,time\_key为"2016-12-23 10:15:00"|
+------------+-------------------------------------------------------------+
| 30分K      | 覆盖10:00:00到10:29:59的分K,time\_key为"2016-12-23 10:30:00"|
+------------+-------------------------------------------------------------+
| 60分K      | 覆盖10:00:00到10:59:59的分K,time\_key为"2016-12-23 11:00:00"|
+------------+-------------------------------------------------------------+

**失败情况**:

1. 股票代码不合法

2. | PLS接口返回错误

获取市场快照 get\_market\_snapshot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_market_snapshot(code_list):

**功能**\ ：一次性获取最近当前股票列表的快照数据（每日变化的信息），该接口对股票个数有限制，一次最多传入200只股票，频率限制每5秒一次

**参数**\ ：

**code\_list**: 股票代码列表，例如，HK.00700，US.AAPL
传入的codelist只允许包含1种股票类型。

**返回**\ ：

ret\_code失败时，ret\_data返回为错误描述字符串；
客户端无符合条件数据时，ret\_code为成功，ret\_data返回None
正常情况下，ret\_data为一个dataframe，其中包括：

**code** ：股票代码；string

**update\_time**\ ： 更新时间(yyyy-MM-dd HH:mm:ss)；string

**last\_price** ： 最新价格；float

**open\_price**\ ： 今日开盘价；float

**high\_price**\ ： 最高价格；float

**low\_price**\ ： 最低价格；float

**prev\_close\_price**\ ： 昨收盘价格；float

**volume**\ ： 成交数量； long

**turnover**\ ： 成交金额；float

**turnover\_rate**\ ： 换手率；float

**suspension**\ ： 是否停牌(True表示停牌)；bool

**listing\_date** ： 上市日期 (yyyy-MM-dd)；string

**circular\_market\_val**\ ： 流通市值；float

**total\_market\_val**: 总市值；float

**wrt\_valid**\ ： 是否是窝轮；bool

**wrt\_conversion\_ratio**: 换股比率；float

**wrt\_type**\ ： 窝轮类型；1=认购证 2=认沽证 3=牛证 4=熊证 string

**wrt\_strike\_price**\ ： 行使价格；float

**wrt\_maturity\_date**: 格式化窝轮到期时间； string

**wrt\_end\_trade**: 格式化窝轮最后交易时间；string

**wrt\_code**: 窝轮对应的正股；string

**wrt\_recovery\_price**: 窝轮回收价；float

**wrt\_street\_vol**: 窝轮街货量；float

**wrt\_issue\_vol**: 窝轮发行量；float

**wrt\_street\_ratio**: 窝轮街货占比；float

**wrt\_delta**: 窝轮对冲值；float

**wrt\_implied\_volatility**: 窝轮引伸波幅；float

**wrt\_premium**: 窝轮溢价；float

**lot\_size**\ ：每手股数；int
                                        
**issued_Shares**：发行股本	int    
                                 
**net_asset**：资产净值  int       
                                       
**net_profit**：盈利（亏损 	int       
                                   
**earning_per_share**：	每股盈利 float  
                                          
**outstanding_shares**：流通股本  int   
                                           
**net_asset_per_share**：每股净资产 float   
                                         
**ey_ratio**：收益率  float      
                                          
**pe_ratio**：市盈率  float     
                                           
**pb_ratio**：市净率  float     
                                       
**price\_spread** ： 当前摆盘价差亦即摆盘数据的买档或卖档的相邻档位的报价差；float    

返回DataFrame，包含上述字段

**窝轮类型** wrt\_type，（字符串类型）：

+------------+--------------------------+
| 窝轮类型   | 标识                     |
+============+==========================+
| "CALL"     | 认购证                   |
+------------+--------------------------+
| "PUT"      | 认沽证                   |
+------------+--------------------------+
| "BULL"     | 牛证                     |
+------------+--------------------------+
| "BEAR"     | 熊证                     |
+------------+--------------------------+
| "N/A"      | 未知或服务器没相关数据   |
+------------+--------------------------+

返回数据量不一定与codelist长度相等， 用户需要自己判断

**调用频率限制：** **5s一次**

**失败情况**:

1. Codelist中股票代码不合法

2. Codelist长度超过规定数量

3. 客户端内部或网络错误

4. 传入的codelist包含多种股票类型

获取分时数据 get\_rt\_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_rt_data(code)

**功能**\ ：获取指定股票的分时数据

**参数**\ ：

**code**: 股票代码，例如，HK.00700，US.APPL

**返回**\ ：

ret\_code失败时，ret\_data返回为错误描述字符串；
客户端无符合条件数据时，ret\_code为成功，ret\_data返回None
正常情况下，ret\_data为一个dataframe，其中包括：

**code**: 股票代码： string

**time**\ ：时间；string

**data\_status**\ ：数据状态；bool，正确为True，伪造为False

**opened\_mins**: 开盘多少分钟：int

**cur\_price**\ ：当前价格：float

**last\_close**: 昨天收盘的价格，float

**avg\_price**: 平均价格，float

**volume**: 成交量，float

**turnover**: 成交额，float

**失败情况**\ ：

1. code不合法

2. | code是未对RT\_DATA类型订阅的股票

3. 客户端内部或网络错误

获取板块集合下的子板块列表 get\_plate\_list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_plate_list(market, plate_class)

**功能**\ ： 获取板块集合下的子板块列表

**参数**\ ：

**market**\ ：市场标识，注意这里不区分沪，深,输入沪或者深都会返回沪深市场的子板块（这个是和客户端保持一致的）

**plate\_class** ：板块分类, string; 例如，"ALL", "INDUSTRY"

**板块分类类型** ，（字符串类型）：

+--------------+------------+
| 板块分类     | 标识       |
+==============+============+
| "ALL"        | 所有板块   |
+--------------+------------+
| "INDUSTRY"   | 行业分类   |
+--------------+------------+
| "REGION"     | 地域分类   |
+--------------+------------+
| "CONCEPT"    | 概念分类   |
+--------------+------------+

**返回**\ ：

| ret\_code失败时，ret\_data返回为错误描述字符串；
  客户端无符合条件数据时，ret\_code为成功，返回None

正常情况下返回K线数据为一个DataFrame包含:

**code**\ ： 板块代码；string

**plate\_name**\ ： 板块名称；string

**plate\_id**: 板块ID：string

港股美股市场的地域分类数据暂时为空

**失败情况**\ ：

1. 市场标识不合法

2. 板块分类不合法

3. 客户端内部或网络错误

获取板块下的股票列表 get\_plate\_stock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_plate_stock(plate_code)

**功能**\ ：获取特定板块下的股票列表,注意这里不区分沪，深,输入沪或者深都会返回沪深市场

**参数**\ ：

**plate\_code**: 板块代码, string,
例如，”SH.BK0001”，”SH.BK0002”，先利用获取子版块列表函数获取子版块代码

**返回**\ ：

ret\_code失败时，ret\_data返回为错误描述字符串；
客户端无符合条件数据时，ret\_code为成功，ret\_data返回None

正常情况下，ret\_data为一个dataframe，其中包括：

**code**\ ：；股票代码：string，例如： ”SZ.000158”，“SZ.000401”

**lot\_size**\ ：每手股数；int

**stock\_name**\ ：股票名称；string，例如： ”天然乳品”，”大庆乳业”

**owner\_market**: 所属股票的市场，仅支持窝轮，其他为空，string

**stock\_child\_type**:
股票子类型；仅支持窝轮，其他为0，string，例如："BEAR"，"BULL"

**stock\_type**\ ：股票类型：string, 例如，"BOND", "STOCK"

+-------------+------------+
| 股票类型    | 标识       |
+=============+============+
| "STOCK"     | 正股       |
+-------------+------------+
| "IDX"       | 指数       |
+-------------+------------+
| "ETF"       | ETF基金    |
+-------------+------------+
| "WARRANT"   | 涡轮牛熊   |
+-------------+------------+
| "BOND"      | 债券       |
+-------------+------------+

**股票子类型** wrt\_type，（字符串类型）：

+--------------+--------------------------+
| 股票子类型   | 标识                     |
+==============+==========================+
| "CALL"       | 认购证                   |
+--------------+--------------------------+
| "PUT"        | 认沽证                   |
+--------------+--------------------------+
| "BULL"       | 牛证                     |
+--------------+--------------------------+
| "BEAR"       | 熊证                     |
+--------------+--------------------------+
| "N/A"        | 未知或服务器没相关数据   |
+--------------+--------------------------+

**失败情况**\ ：

1. | 市场或板块代码不合法，或者该板块不存在

2. 客户端内部或网络错误

获取经纪队列 get\_broker\_queue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, bid_data, ask_data = quote_ctx.get_broker_queue(code)

**功能**\ ：获取股票的经纪队列

**参数**\ ：

**code**: 股票代码, string, 例如，”HK.00700”

**返回**\ ：

ret\_code失败时，bid\_data返回为错误描述字符串；
客户端无符合条件数据时，ret\_code为成功，ask\_data, bid\_data返回None
正常情况下，ask\_data, bid\_data均为dataframe，
其中bid\_data是买盘的数据，包括：

**bid\_broker\_id**: 经纪买盘id；int

**bid\_broker\_name**\ ：经纪买盘名称；string，例如： ”高盛”，”法巴”

**bid\_broker\_pos**: 经纪档位：int, 例如：0，1

其中ask\_data是卖盘的数据，包括：

**ask\_broker\_id**\ ：经纪卖盘id；int

**ask\_broker\_name**\ ：经纪卖盘名称；string，例如： ”高盛”，”法巴”

**ask\_broker\_pos**: 经纪档位：int, 例如：0，1

**失败情况**\ ：

1. 股票代码不合法，不存在

2. 客户端内部或网络错误

获取牛牛程序全局状态 get\_global\_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, state_dict = quote_ctx.get_global_state() 

**功能**\ ：获取牛牛程序全局状态

**参数**\ ：无

| **返回**\ ： ret\_code失败时，ret\_data返回为错误描述字符串；
  客户端无符合条件数据时，ret\_code为成功，ret\_data返回None

正常情况下，ret\_data为dict，包括：

| **Trade\_Logined**: 是否登陆交易服务器,int(0\|1), 1
  表示登陆，0表示未登陆

**Quote\_Logined**\ ：是否登陆行情服务器,int(0\|1), 1
表示登陆，0表示未登陆

**Market\_HK**: 港股主板市场状态,int,字段定义详见下表

**Market\_US**: 美股Nasdaq市场状态,int, 字段定义详见下表

**Market\_SH**: 沪市状态,int,字段定义详见下表

**Market\_SZ**: 深市状态,int,字段定义详见下表

**Market\_HKFuture**: 港股期货市场状态,int,字段定义详见下表

**市场字段说明** ：

+------------+-------------------------------------------+
| 市场状态   | 说明                                      |
+============+===========================================+
| 0          | 未开盘                                    |
+------------+-------------------------------------------+
| 1          | 竞价交易（港股）                          |
+------------+-------------------------------------------+
| 2          | 早盘前等待开盘（港股）                    |
+------------+-------------------------------------------+
| 3          | 早盘（港股）                              |
+------------+-------------------------------------------+
| 4          | 午休（A股、港股）                         |
+------------+-------------------------------------------+
| 5          | 午盘（A股、港股）/ 盘中（美股）           |
+------------+-------------------------------------------+
| 6          | 交易日结束（A股、港股）/ 已收盘（美股）   |
+------------+-------------------------------------------+
| 8          | 盘前开始（美股）                          |
+------------+-------------------------------------------+
| 9          | 盘前结束（美股）                          |
+------------+-------------------------------------------+
| 10         | 盘后开始（美股）                          |
+------------+-------------------------------------------+
| 11         | 盘后结束（美股）                          |
+------------+-------------------------------------------+
| 12         | 内部状态，用于交易日切换                  |
+------------+-------------------------------------------+
| 13         | 夜市交易中（港期货）                      |
+------------+-------------------------------------------+
| 14         | 夜市收盘（港期货）                        |
+------------+-------------------------------------------+
| 15         | 日市交易中（港期货）                      |
+------------+-------------------------------------------+
| 16         | 日市午休（港期货）                        |
+------------+-------------------------------------------+
| 17         | 日市收盘（港期货）                        |
+------------+-------------------------------------------+
| 18         | 日市等待开盘（港期货）                    |
+------------+-------------------------------------------+
| 19         | 港股盘后竞价                              |
+------------+-------------------------------------------+

**失败情况**\ ：

1. 客户端内部或网络错误

订阅 subscribe
~~~~~~~~~~~~~~~~

.. code:: python

    ret_code,ret_data= quote_ctx.subscribe(stock_code, data_type, push=False) 

**功能**\ ：订阅注册需要的实时信息，指定股票和订阅的数据类型即可：

**参数**\ ：

**stock\_code**: 需要订阅的股票代码

**data\_type**: 需要订阅的数据类型

**push**: 推送选项，默认不推送

**返回**\ ：

ret\_code失败时，ret\_data返回为错误描述字符串；
ret\_code为成功，ret\_data返回None 如果指定内容已订阅，则直接返回成功

**失败情况**:

1. 股票代码不合法，不存在

2. 数据类型不合法

3. 订阅额度已满，参考订阅额度表

4. 客户端内部或网络错误

退订 unsubscribe
~~~~~~~~~~~~~~~~~~~

.. code:: python 

    ret_code,ret_data = quote_ctx.unsubscribe(stock_code, data_type, unpush=True)  

ret\_code,ret\_data = unsubscribe(stock\_code, data\_type, unpush=True)

**功能**\ ：退订注册的实时信息，指定股票和订阅的数据类型即可

**参数**\ ：

**stock\_code**: 需要退订的股票代码

**data\_type**: 需要退订的数据类型

**返回**\ ：

ret\_code失败时，ret\_data返回为错误描述字符串；
ret\_code为成功，ret\_data返回None 如果指定内容已退订，则直接返回成功

**失败情况**:

1. 股票代码不合法，不存在

2. 数据类型不合法

3. 内容订阅后不超过60s，就退订

4. 客户端内部或网络错误

查询订阅 query\_subscription
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_data = quote_ctx.query_subscription(query=0) 

**功能**\ ：查询已订阅的实时信息

**参数**\ ：

**query**: 需要查询的类型，int,
0表示当前查询的socket,非0表示查询所有socket的订阅状态

**返回**\ ：

字典类型，已订阅类型为主键，值为订阅该类型的股票，例如

.. code:: python

    { ‘QUOTE’: [‘HK.00700’, ‘SZ.000001’]
      ‘TICKER’: [‘HK.00700’]
      ‘K_1M’: [‘HK.00700’]
      #无股票订阅摆盘和其它类型分K
    }

**失败情况**:

客户端内部或网络错误

获取报价 get\_stock\_quote 和 StockQuoteHandlerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于同步请求使用\ **get\_stock\_quote**\ 直接得到报价

.. code:: python

    ret_code, ret_data = quote_ctx.get_stock_quote(code_list)

**功能**\ ：获取订阅股票报价的实时数据，有订阅要求限制

**参数**\ ：

**code\_list**: 股票代码列表，例如，HK.00700，US.AAPL
传入的codelist只允许包含1种股票类型的股票。
必须确保code\_list中的股票均订阅成功后才能够执行

**返回**\ ： ret\_code失败时，ret\_data返回为错误描述字符串；
客户端无符合条件数据时，ret\_code为成功，返回None
正常情况下，ret\_data为一个dataframe，其中包括：

**code** ：股票代码；string

**data\_date**: 日期： str

**data\_time**: 时间：str

**last\_price** ： 最新价格；float

**open\_price**\ ： 今日开盘价；float

**high\_price**\ ： 最高价格；float

**low\_price**\ ： 最低价格；float

**prev\_close\_price**\ ： 昨收盘价格；float

**volume**\ ： 成交数量； long

**turnover**\ ： 成交金额；float

**turnover\_rate**\ ： 换手率；float

**amplitude** : 振幅：int

**suspension**\ ： 是否停牌(True表示停牌)；bool

**listing\_date** ： 上市日期 (yyyy-MM-dd)；string

**price\_spread** ： 当前价差亦即摆盘数据的买档或卖档的相邻档位的报价差；float

**失败情况**:

1. codelist中股票代码不合法

2. codelist包含未对QUOTE类型订阅的股票

3. 客户端内部或网络错误

4. 传入的codelist包含多种股票类型

对于异步推送数据需要使用\ **StockQuoteHandlerBase**\ 以及继承它的子类处理。例如：

.. code:: python

        class StockQuoteTest(StockQuoteHandlerBase):
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(StockQuoteTest,self).on_recv_rsp(rsp_str) # 基类的on_recv_rsp方法解包返回了报价信息，格式与get_stock_quote一样
                if ret_code != RET_OK:
                    print("StockQuoteTest: error, msg: %s" % content)
                    return RET_ERROR, content
                    
                print("StockQuoteTest ", content) # StockQuoteTest自己的处理逻辑
                
                return RET_OK, content

获取逐笔 get\_rt\_ticker 和 TickerHandlerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于同步请求使用\ **get\_rt\_ticker**\ 直接得到逐笔

.. code:: python

    ret_code, ret_data = quote_ctx.get_rt_ticker(code, num=500)

**功能**\ ： 获取指定股票的实时逐笔。取最近num个逐笔，

**参数**\ ：

**code**: 股票代码，例如，HK.00700，US.AAPL

**num**: 最近ticker个数(有最大个数限制，最近500个）

**返回**\ ：

ret\_code失败时，ret\_data为错误描述字符串；
客户端无符合条件数据时，ret为成功，ret\_data返回None
通常情况下，返回DataFrame，DataFrame每一行是一个逐笔记录，包含：

**stock\_code** 股票代码

**sequence** 逐笔序号

**time** 成交时间；string

**price** 成交价格；double

**volume** 成交数量（股数）；int

**turnover** 成交金额；double

**ticker\_direction** 逐笔方向；int

ticker\_direction:

+---------------+----------+
| 逐笔标识      | 说明     |
+===============+==========+
| TT\_BUY       | 外盘     |
+---------------+----------+
| TT\_ASK       | 内盘     |
+---------------+----------+
| TT\_NEUTRAL   | 中性盘   |
+---------------+----------+

返回的逐笔记录个数不一定与num指定的相等，客户端只会返回自己有的数据。

**失败情况**\ ：

1. code不合法

2. code是未对TICKER类型订阅的股票

3. 客户端内部或网络错误

对于异步推送数据需要使用\ **TickerHandlerBase**\ 以及继承它的子类处理。例如：

.. code:: python

        class TickerTest(TickerHandlerBase):
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(TickerTest,self).on_recv_rsp(rsp_str) # 基类的on_recv_rsp方法解包返回了逐笔信息，格式与get_rt_ticker一样
                if ret_code != RET_OK:
                    print("TickerTest: error, msg: %s" % content)
                    return RET_ERROR, content
                print("TickerTest", content)  # StockQuoteTest自己的处理逻辑
                return RET_OK, content

获取实时K线 get\_cur\_kline 和 CurKlineHandlerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于同步请求使用\ **get\_cur\_kline**\ 直接得到实时K线

.. code:: python

    ret_code, ret_data = quote_ctx.get_cur_kline(code, num, ktype='K_DAY', autype='qfq')

**功能**\ ： 实时获取指定股票最近num个K线数据，最多1000根

**参数**\ ：

**code** 股票代码

**ktype** k线类型，和get\_history\_kline一样

**autype**
复权类型，string；qfq-前复权，hfq-后复权，None-不复权，默认为qfq

对于实时K线数据最多取最近1000根

**返回**\ ：

ret\_code失败时，ret\_data为错误描述字符串；
客户端无符合条件数据时，ret为成功，ret\_data返回None
通常情况下，返回DataFrame，DataFrame内容和get\_history\_kline一样

**失败情况**\ ：

1. code不合法

2. 该股票未对指定K线类型订阅

3. 客户端内部或网络错误

对于异步推送数据需要使用\ **CurKlineHandlerBase**\ 以及继承它的子类处理。例如：

.. code:: python

        class CurKlineTest(CurKlineHandlerBase):
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(CurKlineTest,self).on_recv_rsp(rsp_str) # 基类的on_recv_rsp方法解包返回了实时K线信息，格式除了与get_cur_kline所有字段外，还包含K线类型k_type
                if ret_code != RET_OK:
                    print("CurKlineTest: error, msg: %s" % content)
                    return RET_ERROR, content
                print("CurKlineTest", content) # CurKlineTest自己的处理逻辑
                return RET_OK, content            

获取摆盘 get\_order\_book 和 OrderBookHandlerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于同步请求使用\ **get\_order\_book**\ 直接得到摆盘

.. code:: python

    ret_code, ret_data = quote_ctx.get_order_book(code) 

**功能**\ ：获取实时摆盘数据

**参数**\ ：

**code**: 股票代码，例如，HK.00700，US.AAPL

**返回**\ ： ret\_code失败时，ret\_data为错误描述字符串；
客户端无符合条件数据时，ret为成功，ret\_data返回None
通常情况下，返回字典

.. code:: python

    {‘stock_code’: stock_code
     ‘Ask’:[ (ask_price1, ask_volume1，order_num), (ask_price2, ask_volume2, order_num),…]
    ‘Bid’: [ (bid_price1, bid_volume1, order_num), (bid_price2, bid_volume2, order_num),…]
    }

**失败情况**\ ：

1. code不合法

2. 该股票未对ORDER\_BOOK类型订阅

3. 客户端内部或网络错误

对于异步推送数据需要使用\ **OrderBookHandlerBase**\ 以及继承它的子类处理。例如：

.. code:: python


        class OrderBookTest(OrderBookHandlerBase):
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(OrderBookTest,self).on_recv_rsp(rsp_str) # 基类的on_recv_rsp方法解包返回摆盘信息，格式与get_order_book一样
                if ret_code != RET_OK:
                    print("OrderBookTest: error, msg: %s" % content)
                    return RET_ERROR, content
                print("OrderBookTest", content) # OrderBookTest自己的处理逻辑
                return RET_OK, content            
                

获取分时数据 get\_rt\_data 和 RTDataHandlerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于同步请求使用\ **get\_rt\_data**\ 直接得到分时数据

.. code:: python

    ret_code, ret_data = quote_ctx.get_rt_data(code) 

**功能**\ ：获取实时分时数据

**参数**\ ：

**code**: 股票代码，例如，HK.00700，US.AAPL

**返回**\ ：

ret\_code失败时，ret\_data为错误描述字符串；
客户端无符合条件数据时，ret为成功，ret\_data返回None 通常情况下，返回

**time** 时间

**data\_status** 数据状态

**opened\_mins** 开盘多少分钟

**cur\_price** 当前价格

**last\_close** 昨天收盘的价格

**avg\_price** 平均价格

**volume** 成交量

**turnover** 成交额

**失败情况**\ ：

1. code不合法

2. 该股票未对RT\_DATA类型订阅

3. 客户端内部或网络错误

对于异步推送数据需要使用\ **RTDataHandlerBase**\ 以及继承它的子类处理。例如：

.. code:: python


        class RTDataTest(RTDataHandlerBase):
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(RTDataTest,self).on_recv_rsp(rsp_str) # 基类的on_recv_rsp方法解包返回分时数据，格式与get_rt_data一样
                if ret_code != RET_OK:
                    print("RTDataTest: error, msg: %s" % content)
                    return RET_ERROR, content
                print("RTDataTest", content) 
                return RET_OK, content            
                

获取经纪队列 get\_broker\_queue 和 BrokerHandlerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于同步请求使用\ **get\_broker\_queue**\ 直接得到经纪队列

.. code:: python

    ret_code, ret_data = quote_ctx.get_broker_queue(code)  

**功能**\ ：获取实时经纪队列

**参数**\ ：

**code**: 股票代码，例如，HK.00700，US.AAPL

**返回**\ ：

ret\_code失败时，ret\_data为错误描述字符串；
客户端无符合条件数据时，ret为成功，ret\_data返回None 通常情况下，返回
bid\_data是买盘的数据，包括：

**bid\_broker\_id** 经纪卖盘id

**bid\_broker\_name** 经纪卖盘名称

**bid\_broker\_pos** 经纪档位

ask\_data是卖盘的数据

**ask\_broker\_id** 经纪买盘id

**ask\_broker\_name** 经纪买盘名称

**ask\_broker\_pos** 经纪档位

**失败情况**\ ：

1. code不合法

2. 该股票未对BROKER类型订阅

3. 客户端内部或网络错误

对于异步推送数据需要使用\ **BrokerHandlerBase**\ 以及继承它的子类处理。例如：

.. code:: python


        class BrokerTest(BrokerHandlerBase):
            def on_recv_rsp(self, rsp_str):
                ret_code, ask_content, bid_content = super(BrokerTest, self).on_recv_rsp(rsp_str) # 基类的on_recv_rsp方法解包返回经纪队列，格式与get_broker_queue一样
                if ret_code != RET_OK:
                    print("BrokerTest: error, msg: %s %s " % ask_content % bid_content)
                    return RET_ERROR, ask_content, bid_content
                print("BrokerTest", ask_content, bid_content) 
                return RET_OK, ask_content, bid_content
			
获取多支股票多个单点历史K线 get\_multi\_points\_history\_kline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    ret_code, ret_data = quote_ctx.get_multi_points_history_kline(self, codes, dates, fields, ktype='K_DAY', autype='qfq', no_data_mode=KL_NO_DATA_MODE_FORWARD)

**功能**\ ：获取多支股票多个单点历史K线

**参数**\ ：

**codes**: 单个或多个股票 'HK.00700'  or  ['HK.00700', 'HK.00001']

**dates**: 单个或多个日期 '2017-01-01' or ['2017-01-01', '2017-01-02']

**fields**: 单个或多个K线字段类型，指定需要返回的数据 KL_FIELD.ALL or [KL_FIELD.DATE_TIME, KL_FIELD.OPEN]

**ktype**: K线类型

**autype**: 复权类型

**param no_data_mode**: 请求点无数据时，对应的k线数据取值模式

+----------------------------+--------------------------+
| 取值模式                   | 标识                     |
+============================+==========================+
| KL_NO_DATA_MODE_NONE       | 请求点无数据时返回空     |
+----------------------------+--------------------------+
| KL_NO_DATA_MODE_FORWARD    | 请求点无数据时向前返回   |
+----------------------------+--------------------------+
| KL_NO_DATA_MODE_BACKWARD   | 请求点无数据时向后返回   |
+----------------------------+--------------------------+

**返回**\ ：
ret\_code失败时，ret\_data为错误描述字符串；
通常情况下，返回DataFrame，DataFrame每一行是一个逐笔记录，包含：
**code**\ ： 股票代码；string

**data\_valid**\ ： 数据点是否有效，0=无数据，1=请求点有数据，2=请求点无数据，向前取值，3=请求点无数据，向后取值

**time\_point**\ ： 请求点时间 string “YYYY-MM-DD HH:mm:ss”，暂时最多5个以内时间点。

**time\_key**\ ： K线时间 string “YYYY-MM-DD HH:mm:ss”

**open**\ ： 开盘价；double

**high**\ ： 最高价；double

**close**\ ： 收盘价；double

**low**\ ： 最低价：double

**volume**\ ： 成交量；long

**turnover** ：成交额；double

**pe_ratio**：市盈率 ：double

**turnover_rate**:  换手率：double

**change_rate**:   涨跌幅：double

**失败情况**\ ：

1. code不合法

2.请求时间点为空

3.请求时间点过多
