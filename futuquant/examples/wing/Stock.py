# -*- coding: utf-8 -*-
##股票类

from futuquant.open_context import *
import futu_tools as tools
import json

TAG = "Stock"

class Stock(object):

    def __init__(self, quote_ctx, stock_code, market):
        if(len(stock_code.split('.')) != 2):
            raise Exception("code must like 'HK.00001'")
        if(market is None):
            raise Exception("market can not None!!")
        self.market = market
        self.quote_ctx = quote_ctx
        self.stock_code = stock_code
        if self.stock_code in market.stocks:
            self.name = market.stocks[stock_code]["name"]
            self.lot_size = market.stocks[stock_code]["lot_size"]
        else:
            self.name = ""
            self.lot_size = 0
        self.history, self.history_index = self.getHistoryData()
        # self.getAutype()
        # self.getRTData()
        # self.snapshot = self.getSnapshot()
        # self.getBrokerQueue()
        # self.quote = self.getStockQuote()
        # self.getRTTicker(10)
        # self.getCurKline(ktype='K_1M', num=10)
        # self.getOrderBook()

    def toJson(self):
        data = {}
        data["name"] = self.name
        data["stock_code"] = self.stock_code
        data["lot_size"] = self.lot_size
        data["history"] = self.history
        # data["snapshot"] = self.snapshot
        # data["quote"] = self.quote
        data["history_index"] = self.history_index
        return data


    #定阅
    def subscribe(self, data_type):
        ret, data = self.quote_ctx.subscribe(self.stock_code, data_type)
        if ret != 0:
            print(TAG, "get subscribe error!!", ret, data)
            return None
        return True

    #退定
    def unsubscribe(self, data_type):
        ret, data = self.quote_ctx.unsubscribe(self.stock_code, data_type)
        if ret != 0:
            print(TAG, "get unsubscribe error!!", ret, data)
            return None
        return True

    #复权因子
    def getAutype(self):
        ret, data = self.quote_ctx.get_autype_list([self.stock_code])
        if ret != 0 or data is None:
            print(TAG, "get getAutype error!!", ret)
            return None
        autype_data = []
        for i in data.index:
            row = {}
            for col in data.columns:
                row[col] = data.loc[i][col]
            autype_data.append(row)
        # print(json.dumps(autype_data, indent=2))
        return autype_data

    '----------------实时信息---------------'
    #分时数据
    def getRTData(self):
        self.subscribe("RT_DATA")
        ret, data = self.quote_ctx.get_rt_data(self.stock_code)
        if ret != 0 or data is None:
            print(TAG, "get getRTData error!!", ret)
            return None
        rt_data = []
        for i in data.index:
            row = {}
            for col in data.columns:
                row[col] = data.loc[i][col]
            rt_data.append(row)
        # print(json.dumps(rt_data, indent=2))
        return rt_data

    #市场快照
    def getSnapshot(self):
        ret, data = self.quote_ctx.get_market_snapshot([self.stock_code])
        if ret != 0 or data is None:
            print(TAG, "get getSnapshot error!!", ret)
            return None
        snap_data = {}
        for i in data.index:
            for col in data.columns:
                if col in ["volume"]:
                    snap_data[col] = int(data.loc[i][col])
                else:
                    snap_data[col] = data.loc[i][col]
            break
        # print(snap_data)
        return snap_data

    #经济队列
    def getBrokerQueue(self):
        self.subscribe("BROKER")
        ret, bid_data, ask_data = self.quote_ctx.get_broker_queue(self.stock_code)
        if ret != 0 or (bid_data is None and ask_data is None):
            print(TAG, "get getBrokerQueue error!!", ret)
            return None, None
        bid = []
        for i in bid_data.index:
            row = {}
            for col in bid_data.columns:
                row[col] = bid_data.loc[i][col]
            bid.append(row)
        ask = []
        for i in bid_data.index:
            row = {}
            for col in bid_data.columns:
                row[col] = bid_data.loc[i][col]
            ask.append(row)
        # print(json.dumps(bid, indent=2))
        # print(json.dumps(ask, indent=2))
        return bid, ask

    #获取报价
    def getStockQuote(self):
        self.subscribe("QUOTE")
        ret, data = self.quote_ctx.get_stock_quote(self.stock_code)
        if ret != 0 or data is None:
            print(TAG, "get getStockQuote error!!", ret)
            return None
        quote_data = {}
        # print(data.index)
        for i in data.index:
            for col in data.columns:
                quote_data[col] = data.loc[i][col]
            break
        # print(json.dumps(quote_data, indent=2))
        return quote_data

    #获取逐笔
    def getRTTicker(self, num=500):
        self.subscribe("TICKER")
        ret, data = self.quote_ctx.get_rt_ticker(self.stock_code, num)
        if ret != 0 or data is None:
            print(TAG, "get getRTTicker error!!", ret)
            return None
        rt_ticker = []
        for i in data.index:
            row = {}
            for col in data.columns:
                row[col] = data.loc[i][col]
            rt_ticker.append(row)
        # print(json.dumps(rt_ticker, indent=2))
        return rt_ticker

    #实时K线
    def getCurKline(self, ktype="K_1M", num=1000):
        self.subscribe(ktype)
        ret, data = self.quote_ctx.get_cur_kline(self.stock_code, num, ktype)
        if ret != 0 or data is None:
            print(TAG, "get getCurKline error!!", ret)
            return None
        k_data = []
        for i in data.index:
            row = {}
            for col in data.columns:
                row[col] = data.loc[i][col]
            k_data.append(row)
        # print(json.dumps(k_data, indent=2))
        return k_data

    #当前摆盘
    def getOrderBook(self):
        self.subscribe("ORDER_BOOK")
        ret, data = self.quote_ctx.get_order_book(self.stock_code)
        if ret != 0 or data is None:
            print(TAG, "get getOrderBook error!!", ret)
            return None
        # print(json.dumps(data, indent=2))
        return data["Bid"], data["Ask"]

    '----------------历史信息---------------'
    '''
    start_time：开始时间, None时前一周前
    end_time： 结束时间，None时为Now
    ktype： 数据类型, K_1M K_5M K_15M K_30M K_60M K_DAY K_WEEK K_MON. 
    :return json; csv=True return csv
    '''
    #历史数据
    def getHistoryData(self, start_time=None, end_time=None, ktype="K_DAY", csv=False):
        # print(TAG, "getHistoryData", self.stock_code, start_time, end_time)
        if start_time is None:
            start_time = tools.getPreDaysTime(predays=7)
        ret, data = self.quote_ctx.get_history_kline(self.stock_code, start=start_time, end=end_time, ktype=ktype)
        if ret != 0 or data is None:
            print(TAG, "get getHistoryData error!!", ret)
            return None
        if csv:
            return data.to_csv()
        #columns 'code', 'time_key', 'open', 'close', 'high', 'low', 'pe_ratio', 'turnover_rate', 'volume', 'turnover', 'change_rate'
        history = []
        history_index = []
        for i in data.index:
            row = {}
            for col in data.columns:
                if col in ["volume"]:
                    row[col] = int(data.loc[i][col])
                else:
                    row[col] = data.loc[i][col]
            if ktype in ["K_DAY", "K_WEEK", "K_MON"]:
                time_key = tools.string2datetime(data.loc[i]["time_key"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
                row["time_key"] = time_key
                history_index.append(time_key)
            else:
                history_index.append(data.loc[i]["time_key"])
            history.append(row)
        return history, history_index