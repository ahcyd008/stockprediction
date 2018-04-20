# -*- coding: utf-8 -*-

import datetime
import json

TAG = "futu_tools"

## 与股票无关工具类

#把字符串转成datetime
def string2datetime(string, format):
    return datetime.datetime.strptime(string, format)

def today(format='%Y-%m-%d'):
    return datetime.datetime.today().strftime(format)

def getPreDaysTime(predays=7, format='%Y-%m-%d'):
    today = datetime.datetime.today()
    return (today - datetime.timedelta(days=predays)).strftime(format)

def loadJson(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def storeJson(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        f.close()

def storeString(data, filename):
    with open(filename, 'w') as f:
        f.write(data)
##futu 股票相关工具类
