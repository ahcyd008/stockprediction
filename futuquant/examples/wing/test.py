# encoding=utf-8
# import jieba
from futuquant.open_context import *
import futu_tools as tools
import Market
import Stock


# seg_list = jieba.cut("他来到了网易杭研大厦", cut_all=False)
# print("Full Mode: " + "/ ".join(seg_list))  #

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

hk_market = Market.Market(quote_ctx, "HK")
tools.storeJson(hk_market.toJson(), "hk_market.json")

tencent = Stock.Stock(quote_ctx, "HK.00700", hk_market)
csv = tencent.getHistoryData(start_time="2015-01-01", csv=True)
tools.storeString(csv, "tencent.csv")
tools.storeJson(tencent.toJson(), "tencent.json")

guoqi = Stock.Stock(quote_ctx, "HK.800100", hk_market)
csv = guoqi.getHistoryData(start_time="2015-01-01", csv=True)
tools.storeString(csv, "guoqi.csv")
tools.storeJson(guoqi.toJson(), "guoqi.json")

hengshen = Stock.Stock(quote_ctx, "HK.800000", hk_market)
csv = hengshen.getHistoryData(start_time="2015-01-01", csv=True)
tools.storeString(csv, "hengshen.csv")
tools.storeJson(hengshen.toJson(), "hengshen.json")


# history, history_index = tencent.getHistoryData(start_time="2018-01-01")
# tools.storeJson(tencent.toJson(), "tmp.json")
# print(history_index)

# ret, data = quote_ctx.get_stock_basicinfo("HK", stock_type='IDX')
# print(ret, data)

# ret, data = quote_ctx.get_history_kline("HK.800000", start=tools.getPreDaysTime(predays=7))

# hs = Stock.Stock(quote_ctx, "HK.800000", hk_market)
# print(json.dumps(hs.toJson(), indent=2))

ret_code, ret_data = quote_ctx.query_subscription()

print(ret_code, ret_data)

quote_ctx.close()

