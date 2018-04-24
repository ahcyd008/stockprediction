# encoding=utf-8
# import jieba
from futuquant.open_context import *
import futu_tools as tools
import Market
import Stock

if __name__ == '__main__':
    data_dir = "./data"
    tools.check_dir_exist(data_dir, create=True)

    # seg_list = jieba.cut("他来到了网易杭研大厦", cut_all=False)
    # print("Full Mode: " + "/ ".join(seg_list))  #

    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

    hk_market = Market.Market(quote_ctx, "HK")
    tools.storeJson(hk_market.toJson(), data_dir+"/hk_market.json")

    #腾讯
    tencent = Stock.Stock(quote_ctx, "HK.00700", hk_market)
    csv = tencent.getHistoryData(start_time="2015-01-01", csv=True)
    tools.storeString(csv, data_dir+"/tencent.csv")
    tencent.history, tencent.history_index, tencent.key_index = tencent.getHistoryData(start_time="2015-01-01")
    tools.storeJson(tencent.toJson(), data_dir+"/tencent.json")

    #国企指数
    guoqi = Stock.Stock(quote_ctx, "HK.800100", hk_market)
    csv = guoqi.getHistoryData(start_time="2015-01-01", csv=True)
    tools.storeString(csv, data_dir+"/guoqi.csv")
    guoqi.history, guoqi.history_index, guoqi.key_index = guoqi.getHistoryData(start_time="2015-01-01")
    tools.storeJson(guoqi.toJson(), data_dir+"/guoqi.json")

    #恒生指数
    hengshen = Stock.Stock(quote_ctx, "HK.800000", hk_market)
    csv = hengshen.getHistoryData(start_time="2015-01-01", csv=True)
    tools.storeString(csv, data_dir+"/hengshen.csv")
    hengshen.history, hengshen.history_index, hengshen.key_index = hengshen.getHistoryData(start_time="2015-01-01")
    tools.storeJson(hengshen.toJson(), data_dir+"/hengshen.json")

    ret_code, ret_data = quote_ctx.query_subscription()

    print("query_subscription", ret_code, ret_data)

    quote_ctx.close()

