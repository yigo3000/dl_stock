#-*- coding: UTF-8 -*-
#class input_data():获取指定日期、编号的股票的输入向量。具有以下特点：
#1. 如果指定日期不是交易日，input_data=[]
#2.
#class input_data_label(input_data)：额外可以获取label。考虑了以下情况：
#1. 目标卖出日大于今天，还没有数据，所以不能用来训练，需要将input_data=[]。
#2.
#方法：input_data.get_input_data()
#input_data_label.get_input_data(), input_data_label.get_label_data()

#调试记录：
#20170628发现get_hist_data得到的数据是不复权的，这样的ma5，ma10等数据会不准。需要自己计算ma5,ma10.正在写compute_ma
#正在写_load_report
PROFIT_RATE = 0.1 #目标收益率
SOLD_DATE = 30 #从今天算起，卖出的天数。如果今天买，SOLD_DATE之内有达到过PROFIT_RATE，lable为1;否则lable为0
from datetime import date
from datetime import timedelta
import tushare as ts
import pickle
import numpy as np
import logging
#输入指定股票的指定日期，得到用来训练的vector（numpy类型）
class input_data():
    def __init__(self, ticker,date):
        self.ticker = ticker
        self.buy_date = date
        self.buy_date_str = date_to_str(self.buy_date)

        #原始数据
        self.price=[]
        self.volume=[]
        self.index_price=[]
        self.index_volume=[]

        #最终输出的结果，用于训练
        self.input_data = []
        self.label = None

        #指示数据有效性的
        self.data_valid = True
        self.no_week_ma = False #没有日线的20日均线，说明再往前的日子数据都不用找了.
    def get_input_data(self):
        '''
        #取日线行情:
        #data_day_0 = ts.get_k_data(self.ticker, ktype='d', autype='hfq',index=False,start=self.buy_date_str, end=self.buy_date_str)
        data_day_1 = ts.get_hist_data(self.ticker, ktype='D', start=self.buy_date_str,end=self.buy_date_str) #获取日线
        data_week_1 = ts.get_hist_data(self.ticker, ktype='W', start=date_to_str(self.buy_date-timedelta(6)), end=self.buy_date_str)

        #检查buydate是否存在
        if(data_day_1.iloc[0,0] != self.buy_date_str):
            print("数据无效：目标买入日不是交易日："+self.ticker+": "+self.buy_date_str)
            return []
        #价格
        self.price = [data_day_1.iloc[0,2],data_day_1.iloc[0,7],data_day_1.iloc[0,8],data_day_1.iloc[0,9],
                 data_week_1.iloc[0,7],data_week_1.iloc[0,8],data_week_1.iloc[0,9]]
                 #内容：price, ma5，ma10,ma20,w_ma5,w_ma10,w_ma20
        #检查有没有异常值
        if(self._check_data(self.price) is False):
            return []
        price = normalize(np.array(self.price))

        #成交量
        self.volume = [data_day_1.iloc[0,5],data_day_1.iloc[0,10],data_day_1.iloc[0,11],data_day_1.iloc[0,12],
                    data_week_1.iloc[0, 10], data_week_1.iloc[0, 11], data_week_1.iloc[0, 12] ]
                 # 内容：volume, v_ma5，v_ma10,v_ma20,v_w_ma5,v_w_ma10,v_w_ma20
        volume = normalize(np.array(self.volume))
        '''
        #取日线行情:
        self.price, self.volume = self._get_price_and_volume_data(self.ticker,index=False)
        if(self.data_valid is False):
            return []
        #取中小板指数
        self.index_price, self.index_volume = self._get_price_and_volume_data('399005',index=True)
        #将以上数据归一化
        if(self.data_valid):
            nm_price = normalize(np.array(self.price))
            nm_volume = normalize(np.array(self.volume))
            nm_index_price = normalize(np.array(self.index_price))
            nm_index_volume = normalize(np.array(self.index_volume))
        if(self.data_valid is False):
            return []
        '''
        #data_day_0 = ts.get_k_data('zxb', ktype='d', autype='hfq', index=False, start=self.buy_date_str,
        #                           end=self.buy_date_str)
        data_day_1 = ts.get_hist_data('zxb', ktype='D', start=self.buy_date_str, end=self.buy_date_str)
        data_week_1 = ts.get_hist_data('zxb', ktype='W', start=date_to_str(self.buy_date - timedelta(6)),
                                       end=self.buy_date_str)
        self.index_price = [data_day_1.iloc[0,2],data_day_1.iloc[0,7],data_day_1.iloc[0,8],data_day_1.iloc[0,9],
                 data_week_1.iloc[0,7],data_week_1.iloc[0,8],data_week_1.iloc[0,9]]
                 #内容：price, ma5，ma10,ma20,w_ma5,w_ma10,w_ma20
        index_price = normalize(np.array(self.index_price))
        index_volume = [data_day_1.iloc[0,5],data_day_1.iloc[0,10],data_day_1.iloc[0,11],data_day_1.iloc[0,12],
                    data_week_1.iloc[0, 10], data_week_1.iloc[0, 11], data_week_1.iloc[0, 12] ]
        index_volume = normalize(np.array(index_volume))
                 # 内容：volume, v_ma5，v_ma10,v_ma20,v_w_ma5,v_w_ma10,v_w_ma20
        '''
        #财务数据
        self.report = self._get_report_data()
        if(self.data_valid is False):
            return []
        #整合以上数据
        return np.append(np.append(np.append(np.append(nm_price,nm_volume),nm_index_price),nm_index_volume),np.array(self.report))

    def _check_data(self,data):
        for d in data:
            if d<1.0:
                return False # 异常
        return True # 正常
    def _get_report_data(self):
        '''
        report data必须一次性获取全部股票的全部数据。每次下载很慢，所以保存到本地文件中。包括
        code,代码
        name,名称
        esp,每股收益
        eps_yoy,每股收益同比(%)
        bvps,每股净资产
        roe,净资产收益率(%)
        epcf,每股现金流量(元)
        net_profits,净利润(万元)
        profits_yoy,净利润同比(%)
        distrib,分配方案
        report_date,发布日期
        :return: [eps_yoy,roe,profits_yoy]
        '''
        #season = int((self.buy_date.month - 2) / 3)  # 5,6,7看一季报,8910看二季报，11\12\1看三季报，2\3\4看年报
        season = int((self.buy_date.month - 1) / 3)  # 4,5,6看一季报,789看二季报，10\11\12看三季报，1\2\3
        year = self.buy_date.year #type: int
        if season == 0:
            season = 4
            year = year-1#看上一年的4季报
        tmp_report_date,tmp_report = self._load_report(year,season)
        if(tmp_report_date is None):
            #print("数据无效：找不到季报："+self.ticker+": "+self.buy_date_str)
            date_delt=timedelta(1)
        else:
            date_delt = tmp_report_date-self.buy_date
        while(date_delt.days>-180):#直到找出比buy_date更早的季报,但仅限半年之内的。半年前的太久了，不要。
            season=season-1
            if season == 0:
                season = 4
                year = year-1
            tmp_report_date,tmp_report = self._load_report(year,season)
            if(tmp_report_date is not None):
                date_delt = tmp_report_date-self.buy_date
            if(date_delt.days<0):
                break
        #检查数据合理性
        if(date_delt.days<=-180):
            self.data_valid=False
            print("数据无效：找不到近半年的季报："+self.ticker+": "+self.buy_date_str)
        if(tmp_report[0]==0.0 or tmp_report[1]==0.0 or tmp_report[2]==0.0):
            self.data_valid=False
            print("数据无效：找不到同比增长数据："+self.ticker+": "+self.buy_date_str)
        return tmp_report
    def _load_report(self,year,season):
        #从本地或者网上取得report
        #year: int
        #season: int,1,2,3,4
        report_date=None
        eps_yoy,roe,profits_yoy=0.0,0.0,0.0
        try:#本地有记录
            with open('report_'+str(year)+'_'+str(season)+'.pkl', "rb") as f:
                report = pickle.load(f)
        except:#本地没有存过
            with open('report_' + str(year) + '_' + str(season) + '.pkl', "wb") as f:
                report = ts.get_report_data(year, season)  # 获取业绩报表
                pickle.dump(report, f)
        for i in range(report.index.size):
            if(report.iloc[i,0]==self.ticker):
                #公布报告的日期：
                if(season==4):#如果是4季报，发布年份是下一年
                    report_year=year+1
                else:
                    report_year=year
                report_date = date(report_year,int(report.iloc[i,10][0:2]),int(report.iloc[i,10][3:5]))#先看看报告日期.形如'06-16'

                eps_yoy = report.iloc[i,3]/100
                roe = report.iloc[i,5]/100
                profits_yoy = report.iloc[i,8]/100
                break
        return report_date,[eps_yoy,roe,profits_yoy]
    def _get_price_and_volume_data(self,ticker,index):
        start_day_date = date_to_str(self.buy_date-timedelta(40))
        start_week_date = date_to_str(self.buy_date-timedelta(25)*7)
        #取日线行情:
        data_day = ts.get_k_data(ticker, ktype='d', autype='qfq',index=index,start=start_day_date, end=self.buy_date_str)
        #检查buydate是否存在
        if(data_day.iloc[-1,0] != self.buy_date_str):
            print("数据无效：目标买入日不是交易日："+ticker+": "+self.buy_date_str)
            self.data_valid = False
            return [],[]
        #检查够不够20个交易日
        if(data_day.index.size<20):
            print("数据无效：股票上市日子不够，计算不出日MA20："+ticker+": "+self.buy_date_str)
            self.no_week_ma = True
            self.data_valid = False
            return [],[]
        #取周线行情:
        data_week= ts.get_k_data(ticker, ktype='w', autype='qfq',index=index,start=start_week_date, end=self.buy_date_str)
        #检查够不够20个交易日
        if(data_week.index.size<20):
            print("数据无效：股票上市日子不够，计算不出周MA20："+ticker+": "+self.buy_date_str)
            self.no_week_ma = True
            self.data_valid = False
            return [],[]
        #当日price
        price = [data_day.iloc[-1,2]]
        #日均值price
        price += compute_ma(data_day['close'])
        #周均值price
        price += compute_ma(data_week['close'])
        if(self._check_data(price) is False):
            print("数据无效：股价太小"+ticker+": "+self.buy_date_str)
            self.data_valid = False
            return [],[]
        #当日volume
        volume = [data_day.iloc[-1,5]]
        #日均值volume
        volume += compute_ma(data_day['volume'])
        #周均值volume
        volume += compute_ma(data_week['volume'])
        if(self._check_data(volume) is False):
            print("数据无效：成交量太小"+ticker+": "+self.buy_date_str)
            self.data_valid = False
            return [],[]
        return price,volume

class input_data_label(input_data):
    def __init__(self,ticker,date,sold_date=SOLD_DATE,profit_rate=PROFIT_RATE):
        super(input_data_label,self).__init__(ticker,date)
        self.label=0 #  1：盈利目标达到；0：没达到
        self.sold_date = date+timedelta(sold_date)
        self.sold_date_str = date_to_str(self.sold_date)
        self.profit_rate=profit_rate
    def _get_label(self):
        pass
    def get_label_data(self):
        #检查目标卖出日是否大于训练的日子
        if(self.sold_date>date.today()):
            self.input_data=[]
            print("数据无效：目标卖出日还没到，不能用来训练："+self.ticker+": "+self.buy_date_str+": "+self.sold_date_str)
            self.data_valid=False
            return self.label
        #获取未来的价格
        data_day_0 = ts.get_k_data(self.ticker, ktype='d', autype='hfq',index=False,start=self.buy_date_str, end=self.sold_date_str)
        for i in range(len(data_day_0.index)):
            if((data_day_0.iloc[i,2]-data_day_0.iloc[0,2])/data_day_0.iloc[0,2]>self.profit_rate):
                self.label=1
                break
        return self.label


def normalize(input_list):
    #input_list: np.array
    mean = input_list.mean()
    std = input_list.std()
    for i in range(len(input_list)):
        input_list[i] = (input_list[i] - mean) / std
    return input_list
def date_to_str(date_python):#
    return date_python.strftime('%Y-%m-%d')
def compute_ma(input_data):
    #计算均值，输出list: [ma5,ma10,ma20]
    #以input_data的最后一个值为起点，往左找5天计算ma5，往左找10天计算ma10......
    tmp_sum=0.0
    for i in range(20):
        tmp_sum += input_data[input_data.index[-i-1]]
        if(i==4):
            ma5=tmp_sum/5.0
        if(i==9):
            ma10=tmp_sum/10.0
        if(i==19):
            ma20=tmp_sum/20.0
    return [ma5,ma10,ma20]

def main():
    one_date = date(2017,month=5,day=2)
    one_ticker = '601600'
    one_data = input_data_label(one_ticker,one_date)
    x = one_data.get_input_data()
    y = one_data.get_label_data()

def test():
    logger = logging.getLogger("mylog")
    formatter = logging.Formatter('%(name)-12s %(asctime)s %(levelname)-8s %(message)s', '%a, %d %b %Y %H:%M:%S',)
    file_handler = logging.FileHandler("test_log.txt",encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    zxb_tickers = ts.get_sme_classified()
    one_dateee = date(2017,month=4,day=1)
    for i in range(zxb_tickers.index.size):
        one_ticker = zxb_tickers.iloc[i,0]
        for j in range(3650):
            one_data = input_data_label(one_ticker,one_dateee)
            x = one_data.get_input_data()
            y = one_data.get_label_data()

            #检查数据合理性
            if(one_data.data_valid):
                logger.debug('valid: '+str(one_ticker)+'_'+date_to_str(one_dateee)+"_"+str(x)+'_'+str(y)) #数据有效
            elif(one_data.no_week_ma):
                logger.debug('invalid: '+str(one_ticker)+'_'+date_to_str(one_dateee)+"_"+str(x)+'_'+str(y))
                break  #数据无效，而且再往前的日子的数据都不会有效了
            one_dateee = one_dateee-timedelta(1)

if __name__ == '__main__':
    #main()
    test()

#tips:
#判断未来的涨幅，应该使用后复权的未来股价与设定日的股价相减。
#即：设定日-》-》-》未来股价（后复权）
#后复权：就是把除权后的价格按以前的价格换算过来。复权后以前的价格不变，现在的价格增加，所以为了利于分析一般推荐前复权。
