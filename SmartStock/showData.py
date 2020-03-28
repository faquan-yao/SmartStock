import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

def formatDates(date):
    return str(date)

def showData(dates, values, AbscissaSize=10):
    fig = plt.figure(figsize=(24, 12))  # 调整画图空间的大小
    plt.plot(dates, values, linestyle='-', marker='*', c='r', alpha=0.5)  # 作图
    ax = plt.gca()

    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))  # 设定x轴主要格式
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(int(len(dates) / AbscissaSize)))  # 设定坐标轴的显示的刻度间隔
    fig.autofmt_xdate()  # 防止x轴上的数据重叠，自动调整。
    plt.savefig("/home/faquanyao/work_space/stocks/data/000001.SZ.png")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("/home/faquanyao/work_space/stocks/data/000001.SZ.csv")

    dates = df['trade_date']
    print(dates)
    values = df['close']

    showData(dates, values, 10)


