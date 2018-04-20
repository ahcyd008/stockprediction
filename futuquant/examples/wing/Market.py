# -*- coding: utf-8 -*-
##市场类 HK US SH SZ HK_FUTURE

from futuquant.open_context import *
import futu_tools as tools

TAG = "Market"

class Market(object):

    def __init__(self, quote_ctx, market):
        self.market = market
        self.quote_ctx = quote_ctx
        self.stocks = self.getStocks()
        self.idxs = self.getIdxs()

    def toJson(self):
        data = {}
        data["market"] = self.market
        data["stocks"] = self.stocks
        data["idxs"] = self.idxs
        return data

    # 基本信息
    def getStocks(self):
        ret, data = self.quote_ctx.get_stock_basicinfo(self.market, stock_type='STOCK')
        if ret != 0 or data is None:
            print(TAG, "get getStocks error!!", ret)
            return None
        #code', 'name', 'lot_size', 'stock_type', 'stock_child_type', 'owner_stock_code', 'listing_date', 'stockid'
        stocks = {}
        for i in data.index:
            row = {
                "code": data.loc[i]["code"],
                "name": data.loc[i]["name"],
                "lot_size": int(data.loc[i]["lot_size"]),
                "listing_date": data.loc[i]["listing_date"]
            }
            stocks[data.loc[i]["code"]] = row
        return stocks

    # 基本信息
    def getIdxs(self):
        ret, data = self.quote_ctx.get_stock_basicinfo(self.market, stock_type='IDX')
        if ret != 0 or data is None:
            print(TAG, "get getIdxs error!!", ret)
            return None
        #code', 'name', 'lot_size', 'stock_type', 'stock_child_type', 'owner_stock_code', 'listing_date', 'stockid'
        idxs = {}
        for i in data.index:
            row = {
                "code": data.loc[i]["code"],
                "name": data.loc[i]["name"],
                "lot_size": int(data.loc[i]["lot_size"]),
                "listing_date": data.loc[i]["listing_date"]
            }
            idxs[data.loc[i]["code"]] = row
        return idxs