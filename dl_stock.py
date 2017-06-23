#-*- coding: UTF-8 -*-
#运行 mpiexec.exe -np 16 python mytrade.py
'''
#macro
HS300_SUB_RAISE=0.1 #指标：沪深300涨幅减去个股涨幅大于10%
REBOUND_FROM_LOWEST=0.1
VOLUME_RAISE_FROM_LOWEST=0.5
def get_ultra_plunge(df_start_date,df_end_date):
    PLUNGE_RANGE=0.10
    pass

def get_long_one():
    pass

def get_raise_rate_of_one_stock(code,start_date,end_date,ktype="D"):
    import tushare as ts
    df = ts.get_hist_data(code,start=start_date,end=end_date,ktype=ktype)
    if(len(df)>0):
        _price_open=df["open"][len(df)-1]
        _price_close=df["close"][0]
        _raise=(_price_close-_price_open)/_price_open
        _price_lowest=99999
        _volume_close=df["volume"][0]#结束日的成交量
        _volume_lowest=9999999999
        for i in range(len(df)):
            if(df["low"][i]<_price_lowest):
                _price_lowest=df["low"][i]
            if(df["volume"][i]<_volume_lowest):
                _volume_lowest=df["volume"][i]
    else:
        return 0,0,0,0,0,0
    return _raise,_price_open,_price_close,_price_lowest,_volume_close,_volume_lowest

if __name__ == "__main__":
    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()#my rank()my position)
    comm_size = comm.Get_size()#number of total work items
    import tushare as ts
    import pickle

    with open('yejis.pkl','rb') as f:#从文件获取业绩报表
        yejis = pickle.load(f)

    yejis_sort=yejis.sort(columns="eps")#按照eps列由小到大排序
    #print(yejis_sort)
    index_of_profitable=0
    for i in range(len(yejis_sort)):#找到eps（每股收益）是正值的点
        tmp=yejis_sort.iloc[i,2]
        if(yejis_sort.iloc[i,2]>0):
            index_of_profitable=i
            break
    #删除不盈利的
    yeji_profit=yejis_sort[index_of_profitable:]

    start_date="2016-07-16"
    end_date="2016-08-01"

    hs300_raise,hs300_open,hs300_close,hs300_lowest,hs300_volume,hs300_volume_lowest=get_raise_rate_of_one_stock("hs300",start_date,end_date)

#step 1: 寻找过于低估的股票。这里开始多个线程做不同的工作
    from pandas import *
    result=DataFrame(columns=["code","NAME","raise","open","close"],index=[0])
    num_of_stocks_per_thread=len(yeji_profit)//comm_size +1
    for i in range(0,num_of_stocks_per_thread):
        #if(i==15):
            #print(yeji_profit.iloc[_stock_index,0])
        _stock_index=comm_rank*num_of_stocks_per_thread+i
        if(_stock_index>=len(yeji_profit)):
            break
        tmp_raise,tmp_open,tmp_close,tmp_lowest,tmp_volume,tmp_volume_lowest=get_raise_rate_of_one_stock(yeji_profit.iloc[_stock_index,0],start_date,end_date)
        #if hs300_raise-tmp_raise>HS300_SUB_RAISE:
        #if hs300_raise-tmp_raise>HS300_SUB_RAISE and (tmp_close-tmp_lowest)/tmp_lowest<0.05:#找到跌幅比沪深300多10%，且从最低点反弹小于5%的
        #if hs300_raise-tmp_raise>HS300_SUB_RAISE and (tmp_close-tmp_lowest)/tmp_lowest<0.10 and tmp_raise>-0.20:#找到跌幅比沪深300多10%，且从最低点反弹小于5%的
        if hs300_raise-tmp_raise>HS300_SUB_RAISE and (tmp_close-tmp_lowest)/tmp_lowest<REBOUND_FROM_LOWEST and tmp_raise>-0.20 and (tmp_volume-tmp_volume_lowest)/tmp_volume_lowest<VOLUME_RAISE_FROM_LOWEST:#找到跌幅比沪深300多10%，且从最低点反弹小于5%的
            result.ix[_stock_index]=Series([yeji_profit.iloc[_stock_index,0],yeji_profit.iloc[_stock_index,1],tmp_raise,tmp_open,tmp_close],index=result.columns)

        #result.ix[len(result)]=Series([yeji_profit.iloc[i,0],yeji_profit.iloc[i,1],tmp_raise,tmp_open,tmp_close],index=result.columns)
        #print(i)


#各个线程把结果发给rank0，由rank0合并、输出结果
    if(comm_rank!=0):
        comm.send(result.drop(0),dest=0)
    else:
        for i in range(comm_size-1):
            result=concat([result, comm.recv(source=i + 1)])
        print(result.drop(0))
    print("process %s finished." %comm_rank)
    result.to_csv("result.csv")
    #print(result)
'''

PROFIT_RATE = 0.1 #目标收益率
SOLD_DATE = 30 #从今天算起，卖出的天数。如果今天买，SOLD_DATE之内有达到过PROFIT_RATE，lable为1;否则lable为0
from datetime import date
from datetime import timedelta
import tushare as ts
import pickle
import numpy as np
#输入指定股票的指定日期，得到用来训练的vector（numpy类型）
class input_data():
    def __init__(self, ticker,date):
        self.ticker = ticker
        self.buy_date = date
        self.buy_date_str = date_to_str(self.buy_date)
        self.sold_date = date+timedelta(SOLD_DATE)
        self.sold_date_str = date_to_str(self.sold_date)
        self.input_data = []
        self.label = None

        self._get_input_data()
    def _get_input_data(self):
        #取日线行情:
        data_day_0 = ts.get_k_data(self.ticker, ktype='d', autype='hfq',index=False,start=self.buy_date_str, end=self.sold_date_str)
        data_day_1 = ts.get_hist_data(self.ticker, ktype='D', start=self.buy_date_str,end=self.buy_date_str)
        data_week_1 = ts.get_hist_data(self.ticker, ktype='W', start=date_to_str(self.buy_date-timedelta(6)), end=self.buy_date_str)

        #检查buydate是否存在
        if(data_day_0.iloc[0,0] != self.buy_date_str):
            return None
        #价格
        price = [data_day_0.iloc[0,2],data_day_1.iloc[0,7],data_day_1.iloc[0,8],data_day_1.iloc[0,9],
                 data_week_1.iloc[0,7],data_week_1.iloc[0,8],data_week_1.iloc[0,9]]
                 #内容：price, ma5，ma10,ma20,w_ma5,w_ma10,w_ma20
        price = normalize(np.array(price))

        #成交量
        volume = [data_day_0.iloc[0,5],data_day_1.iloc[0,10],data_day_1.iloc[0,11],data_day_1.iloc[0,12],
                    data_week_1.iloc[0, 10], data_week_1.iloc[0, 11], data_week_1.iloc[0, 12] ]
                 # 内容：volume, v_ma5，v_ma10,v_ma20,v_w_ma5,v_w_ma10,v_w_ma20
        volume = normalize(np.array(volume))

        #中小板指数
        data_day_0 = ts.get_k_data('zxb', ktype='d', autype='hfq', index=False, start=self.buy_date_str,
                                   end=self.buy_date_str)
        data_day_1 = ts.get_hist_data('zxb', ktype='D', start=self.buy_date_str, end=self.buy_date_str)
        data_week_1 = ts.get_hist_data('zxb', ktype='W', start=date_to_str(self.buy_date - timedelta(6)),
                                       end=self.buy_date_str)
        index_price = [data_day_0.iloc[0,2],data_day_1.iloc[0,7],data_day_1.iloc[0,8],data_day_1.iloc[0,9],
                 data_week_1.iloc[0,7],data_week_1.iloc[0,8],data_week_1.iloc[0,9]]
                 #内容：price, ma5，ma10,ma20,w_ma5,w_ma10,w_ma20
        index_price = normalize(np.array(index_price))
        index_volume = [data_day_0.iloc[0,5],data_day_1.iloc[0,10],data_day_1.iloc[0,11],data_day_1.iloc[0,12],
                    data_week_1.iloc[0, 10], data_week_1.iloc[0, 11], data_week_1.iloc[0, 12] ]
        index_volume = normalize(np.array(index_volume))
                 # 内容：volume, v_ma5，v_ma10,v_ma20,v_w_ma5,v_w_ma10,v_w_ma20

        #财务数据
        report = self._get_report_data()
        #self.input_data =

    def _get_label(self):
        pass
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
        :return:
        '''
        season = int((self.buy_date.month - 2) / 3)  # 5,6,7看一季报,8910看二季报，11\12\1看三季报，2\3\4看年报
        if season == 0:
            season = 4

    def _load_report(self,season):
        #从本地或者网上取得report
        try:#本地有记录
            with open('report_'+self.buy_date_str[:4]+'_'+str(season)+'.pkl', "rb") as f:
                report = pickle.load(f)
        except:#本地没有存过
            with open('report_' + self.buy_date_str[:4] + '_' + str(season) + '.pkl', "wb") as f:
                report = ts.get_report_data(self.buy_date.year, season)  # 获取业绩报表
                pickle.dump(report, f)
        for i in range(report.index.size):
            if(report.iloc[i,0]==self.ticker):
                report_date = date(self.buy_date.year,int(report.iloc[i,10][0:2]),int(report.iloc[i,10][3:5]))#先看看报告日期.形如'06-16'
                #如果0<report_date-buydate<90，不能用这个report,应该用上一个季度的erport
                eps_yoy = report.iloc[i,3]
                roe = report.iloc[i,5]
                profits_yoy = report.iloc[i,8]
                break
        return


def normalize(input_list):
    #input_list: np.array
    mean = input_list.mean()
    std = input_list.std()
    for i in range(len(input_list)):
        input_list[i] = (input_list[i] - mean) / std
    return input_list
def date_to_str(date_python):#
    return date_python.strftime('%4Y-%2m-%2d')
def main():
    one_date = date(2017,month=6,day=12)
    one_ticker = '601600'
    one_id = input_data(one_ticker,one_date)
if __name__ == '__main__':
    main()

#tips:
#判断未来的涨幅，应该使用后复权的未来股价与设定日的股价相减。
#即：设定日-》-》-》未来股价（后复权）
#后复权：就是把除权后的价格按以前的价格换算过来。复权后以前的价格不变，现在的价格增加，所以为了利于分析一般推荐前复权。
