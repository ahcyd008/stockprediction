# -*- coding: utf-8 -*-
"""
    跟踪止损:跟踪止损是一种更高级的条件单，需要指定如下参数，以便制造出移动止损价
    跟踪止损的详细介绍：https://www.futu5.com/faq/topic214
"""

from math import floor
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.split(os.path.abspath(os.path.pardir))[0])

from futuquant.open_context import *
from futuquant.examples.emailplugin import EmailNotification
from futuquant.examples.stocksell import simple_sell, smart_sell

RET_OK = 0
RET_ERROR = -1


class TrailingStopHandler(StockQuoteHandlerBase):
    """"跟踪止损数据回调类"""

    def __init__(self, quote_ctx, is_hk_trade, method, drop):
        super(StockQuoteHandlerBase, self).__init__()
        self.quote_ctx = quote_ctx
        self.is_hk_trade = is_hk_trade
        self.method = method
        self.drop = drop
        self.finished = False
        self.stop = None
        self.price_lst = []
        self.stop_lst = []
        self.time_lst = []

    def on_recv_rsp(self, rsp_str):
        """数据接收回调函数"""
        ret, content = super(TrailingStopHandler, self).on_recv_rsp(rsp_str)
        if ret != RET_OK:
            print('StockQuote error {}'.format(content))
            return ret, content
        if self.finished:
            return ret, content
        ret, data = self.quote_ctx.get_global_state()
        if ret != RET_OK:
            print('获取全局状态失败')
            trading = False
        else:
            hk_trading = (data['Market_HK'] == '3' or data['Market_HK'] == '5')
            us_trading = (data['Market_US'] == '3')
            trading = hk_trading if self.is_hk_trade else us_trading

        if not trading:
            print('不处在交易时间段')
            return RET_OK, content
        last_price = content.iloc[0]['last_price']

        if self.stop is None:
            self.stop = last_price - self.drop if self.method == 0 else last_price * (1 - self.drop)
        elif (self.stop + self.drop < last_price) if self.method == 0 else (self.stop < last_price * (1 - self.drop)):
            self.stop = last_price - self.drop if self.method == 0 else last_price * (1 - self.drop)
        elif self.stop >= last_price:
            # 交易己被触发
            self.finished = True
            print('交易被触发')

        self.price_lst.append(last_price)
        self.stop_lst.append(self.stop)
        print('last_price is {}, stop is {}'.format(last_price, self.stop))

        return RET_OK, content


def trailing_stop(api_svr_ip='127.0.0.1', api_svr_port=11111, unlock_password="", code='HK.00700',
                  trade_env=1, method=0, drop=1, volume=100, how_to_sell=1, diff=0, rest_time=2,
                  enable_email_notification=False, receiver='haha.email'):
    """
    止损策略函数
    :param api_svr_ip: (string)ip
    :param api_svr_port: (int)port
    :param unlock_password: (string)交易解锁密码, 必需修改! 模拟交易设为一个非空字符串即可
    :param code: (string)股票
    :param trade_env: 0: 真实交易 1: 模拟交易 (美股暂不支持模拟交易)
    :param method: 0: 股票下跌drop价格就会止损  1: 股票下跌drop的百分比就会止损
    :param drop: method == 0, 股票下跌的价格   method == 1，股票下跌的百分比，0.01表示下跌1%则止损
    :param volume: 需要卖掉的股票数量
    :param how_to_sell: 以何种方式卖出股票，默认值为0  0: 以(市价-diff)的价格卖出  1: 以smart_sell方式卖出
    :param diff: 默认为0，当how_to_sell为0时，以(市价-diff)的价格卖出
    :param rest_time: 每隔REST_TIME秒，会检查订单状态, 需要>=2
    :param enable_email_notification: 激活email功能
    :param receiver: 邮件接收者
    """
    EmailNotification.set_enable(enable_email_notification)
    if how_to_sell != 0 and how_to_sell != 1:
        print('how_to_sell must be 0 or 1')
        raise Exception('how_to_sell value error')
    if trade_env != 0 and trade_env != 1:
        print('trade_env must be 0 or 1')
        raise Exception('trade_env value error')
    if method != 0 and method != 1:
        print('method must be 0 or 1')
        raise Exception('method value error')

    quote_ctx = OpenQuoteContext(host=api_svr_ip, port=api_svr_port)
    is_hk_trade = 'HK.' in code
    if is_hk_trade:
        trade_ctx = OpenHKTradeContext(host=api_svr_ip, port=api_svr_port)
    else:
        if trade_env != 0:
            raise Exception('美股不支持仿真环境')
        trade_ctx = OpenUSTradeContext(host=api_svr_ip, port=api_svr_port)

    if unlock_password == "":
        raise Exception('请先配置交易密码')
    if trade_env == 0:
        ret, data = trade_ctx.unlock_trade(unlock_password)
        if ret != RET_OK:
            raise Exception('解锁交易失败')
    ret, data = trade_ctx.position_list_query(envtype=trade_env)
    if ret != RET_OK:
        raise Exception("无法获取持仓列表")

    try:
        qty = data[data['code'] == code].iloc[0]['qty']
    except:
        raise Exception('你没有持仓！无法买卖')
    qty = int(qty)
    if volume == 0:
        volume = qty
    elif volume < 0:
        raise Exception('volume lower than  0')
    else:
        if qty < volume:
            raise Exception('持仓不足')
    if volume <= 0:
        raise Exception('没有持仓')
    ret, data = quote_ctx.get_market_snapshot([code])
    if ret != RET_OK:
        raise Exception('获取lot size失败')
    lot_size = data.iloc[0]['lot_size']
    if volume % lot_size != 0:
        raise Exception('volume 必须是{}的整数倍'.format(lot_size))
    ret, data = quote_ctx.subscribe(code, 'QUOTE', push=True)
    if ret != RET_OK:
        raise Exception('订阅QUOTE错误: error {}:{}'.format(ret, data))
    ret, data = quote_ctx.subscribe(code, 'ORDER_BOOK')
    if ret != RET_OK:
        print('error {}:{}'.format(ret, data))
        raise Exception('订阅order book失败: error {}:{}'.format(ret, data))

    if diff:
        if is_hk_trade:
            ret, data = quote_ctx.get_order_book(code)
            if ret != RET_OK:
                raise Exception('获取order book失败: cannot get order book'.format(data))
            min_diff = round(abs(data['Bid'][0][0] - data['Bid'][1][0]), 3)
            if floor(diff / min_diff) * min_diff != diff:
                raise Exception('diff 应是{}的整数倍'.format(min_diff))
        else:
            if round(diff, 2) != diff:
                raise Exception('美股价差保留2位小数{}->{}'.format(diff, round(diff, 2)))
    if method == 0:
        if is_hk_trade:
            if floor(drop / min_diff) * min_diff != drop:
                raise Exception('drop必须是{}的整数倍'.format(min_diff))
        else:
            if round(drop, 2) != drop:
                raise Exception('drop必须保留2位小数{}->{}'.format(drop, round(drop, 2)))
    elif method == 1:
        if drop < 0 or drop > 1:
            raise Exception('drop must in [0, 1] if method is 1')

    trailing_stop_handler = TrailingStopHandler(quote_ctx, is_hk_trade, method, drop)
    quote_ctx.set_handler(trailing_stop_handler)
    quote_ctx.start()
    while True:
        if trailing_stop_handler.finished:
            # sell the stock
            qty = volume
            sell_price = trailing_stop_handler.stop
            while qty > 0:
                if how_to_sell == 0:
                    data = simple_sell(quote_ctx, trade_ctx, code, sell_price - diff, qty, trade_env)
                else:
                    data = smart_sell(quote_ctx, trade_ctx, code, qty, trade_env)
                if data is None:
                    print('下单失败')
                    EmailNotification.send_email(receiver, '下单失败', '股票代码{}，数量{}'.format(code, volume))
                orderid = data.iloc[0]['orderid']
                envtype = data.iloc[0]['envtype']
                time.sleep(rest_time)
                ret, data = trade_ctx.order_list_query(envtype=envtype)
                if ret != RET_OK:
                    raise Exception('获取订单状态失败')
                status = data[data['orderid'] == orderid].iloc[0]['status']
                dealt_qty = data[data['orderid'] == orderid].iloc[0]['dealt_qty']
                order_price = data[data['orderid'] == orderid].iloc[0]['price']
                status = int(status)
                dealt_qty = int(dealt_qty)
                qty -= dealt_qty

                if status == 3:
                    print('全部成交:股票代码{}, 成交总数{}，价格{}'.format(code, dealt_qty, order_price))
                    EmailNotification.send_email(receiver, '全部成交', '股票代码{}，成交总数{}，价格{}'
                                                 .format(code, dealt_qty, order_price))
                elif status == 2:
                    print('部分成交:股票代码{}，成交总数{}，价格{}'.format(code, dealt_qty, order_price))
                    EmailNotification.send_email(receiver, '部分成交', '股票代码{}，成交总数{}，价格{}'
                                                 .format(code, dealt_qty, order_price))
                    while True:
                        ret, data = trade_ctx.set_order_status(0, orderid=orderid, envtype=trade_env)
                        if ret != RET_OK:
                            time.sleep(rest_time)
                            continue
                        else:
                            break
                else:
                    while True:
                        ret, data = trade_ctx.set_order_status(0, orderid=orderid, envtype=trade_env)
                        if ret != RET_OK:
                            time.sleep(rest_time)
                            continue
                        else:
                            break
                if how_to_sell == 0:
                    ret, data = quote_ctx.get_order_book(code)
                    if ret != RET_OK:
                        raise Exception('获取order_book失败')
                    sell_price = data['Bid'][0][0]

            # draw price and stop
            price_lst = trailing_stop_handler.price_lst
            plt.plot(np.arange(len(price_lst)), price_lst)
            stop_list = trailing_stop_handler.stop_lst
            plt.plot(np.arange(len(stop_list)), stop_list)
            break

    quote_ctx.close()
    trade_ctx.close()


if __name__ == '__main__':
    # 全局参数配置
    API_SVR_IP = '119.29.141.202'
    API_SVR_PORT = 11111
    UNLOCK_PASSWORD = "a"
    CODE = 'HK.00700'  # 'US.BABA' #'HK.00700'
    TRADE_ENV = 1
    METHOD = 0
    DROP = 0.2
    VOLUME = 0
    HOW_TO_SELL = 0
    DIFF = 0.2
    REST_TIME = 2  # 每隔REST_TIME秒，会检查订单状态, 需要>=2

    # 邮件通知参数
    ENABLE_EMAIL_NOTIFICATION = True
    RECEIVER = 'your receive email'

    trailing_stop(API_SVR_IP, API_SVR_PORT, UNLOCK_PASSWORD, CODE, TRADE_ENV, METHOD, DROP,
                  VOLUME, HOW_TO_SELL, DIFF, REST_TIME, ENABLE_EMAIL_NOTIFICATION, RECEIVER)
