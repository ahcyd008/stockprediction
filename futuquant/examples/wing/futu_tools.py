# -*- coding: utf-8 -*-

import datetime
import json
import os

TAG = "futu_tools"

## 与股票无关工具类
def check_dir_exist(dirname, create=False, required=False):
    if not os.path.exists(dirname):
        if required:
            print('dir is required!', dirname)
            exit(1)
        if create:
            os.makedirs(dirname)
            print('create dir:', dirname)
            return True
        return False
    return True

def check_file_exist(filename, required=False):
    if not os.path.exists(filename):
        if required:
            print('filename is required!', filename)
            exit(1)
        print('file not exist:', filename)
        return False
    print('file exist:', filename)
    return True

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

def load_text_lines(filename):
    with open(filename) as txt_file:
        lines = txt_file.readlines()
        return lines
##futu 股票相关工具类
