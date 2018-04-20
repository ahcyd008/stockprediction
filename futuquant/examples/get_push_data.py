# -*- coding: utf-8 -*-
"""
Examples for use the python functions: get push data
"""
from futuquant.open_context import *


class StockQuoteTest(StockQuoteHandlerBase):
    """
    获得报价推送数据
    """

    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, content = super(StockQuoteTest, self).on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print("StockQuoteTest: error, msg: %s" % content)
            return RET_ERROR, content
        print("StockQuoteTest\n ", content)
        return RET_OK, content


class OrderBookTest(OrderBookHandlerBase):
    """
    获得摆盘推送数据
    """

    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, content = super(OrderBookTest, self).on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print("OrderBookTest: error, msg: %s" % content)
            return RET_ERROR, content
        print("OrderBookTest\n", content)
        return RET_OK, content


class CurKlineTest(CurKlineHandlerBase):
    """
    获取K线推送数据
    """

    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, content = super(CurKlineTest, self).on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print("CurKlineTest: error, msg: %s" % content)
            return RET_ERROR, content
        print("CurKlineTest\n", content)
        return RET_OK, content


class TickerTest(TickerHandlerBase):
    """
    获取逐笔推送数据
    """

    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, content = super(TickerTest, self).on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print("TickerTest: error, msg: %s" % content)
            return RET_ERROR, content
        print("TickerTest\n", content)
        return RET_OK, content


class RTDataTest(RTDataHandlerBase):
    """
    获取分时推送数据
    """

    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, content = super(RTDataTest, self).on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print("RTDataTest: error, msg: %s" % content)
            return RET_ERROR, content
        print("RTDataTest\n")
        print(content)
        return RET_OK, content


class BrokerTest(BrokerHandlerBase):
    """
    获取经纪队列推送数据
    """

    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, content = super(BrokerTest, self).on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print("BrokerTest: error, msg: %s " % content)
            return RET_ERROR, content
        print("BrokerTest\n", content[0])
        print("BrokerTest\n", content[1])
        return RET_OK, content


class HeartBeatTest(HeartBeatHandlerBase):
    """
    心跳的推送
    """

    def on_recv_rsp(self, rsp_str):
        """数据响应回调函数"""
        ret_code, timestamp = super(HeartBeatTest, self).on_recv_rsp(rsp_str)
        if ret_code == RET_OK:
            print("heart beat server timestamp = ", timestamp)
        return ret_code, timestamp

if __name__ == "__main__":
    quote_context = OpenQuoteContext(host='127.0.0.1', port=11111)

    print(quote_context.get_global_state())
    quote_context.set_handler(HeartBeatTest())

    # 获取推送数据
    code = 'US.AAPL'
    quote_context.subscribe(code, "QUOTE", push=True)
    quote_context.set_handler(StockQuoteTest())
    quote_context.start()

    ret, data = quote_context.get_stock_quote(code)
    print (data)
    ret, data = quote_context.get_market_snapshot(code)
    print (data)

    sleep(10)
    quote_context.close()
