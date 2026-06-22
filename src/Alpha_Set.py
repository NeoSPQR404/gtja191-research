import pandas as pd
import numpy as np
from Factor_Calculator import Factor_Calculator as FC
from typing import List, Dict, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class Alpha_Set:

    @staticmethod
    def alpha1(data):
        # 提取所需数据列
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        output = (-1 * FC.CORR(
            FC.RANK(FC.DELTA(FC.LOG(volume), 1)),
            FC.RANK((close - open) / open),
            6
        ))
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha2(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        output = (-1 * FC.DELTA(
            ((close - low) - (high - close)) / (high - low),
            1
        ))
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha3(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        cond = close == FC.DELAY(close, 1)
        val_true = 0
        val_false = close - np.where(
            close > FC.DELAY(close, 1),
            np.minimum(low, FC.DELAY(close, 1)),
            np.maximum(high, FC.DELAY(close, 1))
        )
        val = np.where(cond, val_true, val_false)
        output = FC.SUM(val, 6)
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha4(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        ma8 = FC.SUM(close, 8) / 8
        std8 = FC.STD(close, 8)
        ma2 = FC.SUM(close, 2) / 2
        cond1 = (ma8 + std8) < ma2
        cond2 = ma2 < (ma8 - std8)
        vol_ratio = volume/FC.MEAN(volume, 20)
        cond3 = (vol_ratio > 1) | (vol_ratio == 1)
        output = np.where(cond1, -1.0, np.where(cond2, 1.0, np.where(cond3, 1.0, -1.0)))
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha5(data):
        volume = data['volume'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        output = (-1 * FC.TSMAX(
            FC.CORR(FC.TSRANK(volume, 5), FC.TSRANK(high, 5), 5),
            3
        ))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha6(data):
        open = data['open'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        output = FC.RANK(FC.SIGN(FC.DELTA((open * 0.85 + high * 0.15), 4))) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha7(data):
        vwap = data['vwap'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (FC.RANK(FC.TSMAX(vwap - close, 3)) +
                FC.RANK(FC.TSMIN(vwap - close, 3))) * \
            FC.RANK(FC.DELTA(volume, 3))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha8(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        output = FC.RANK(FC.DELTA(((high + low) / 2 * 0.2 + vwap * 0.8), 4) * -1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha9(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = FC.SMA(
            ((high + low) / 2 - (FC.DELAY(high, 1) + FC.DELAY(low, 1)) / 2) *
            (high - low) / volume, 7, 2
        )
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha10(data):
        ret = data['ret'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        inner = np.where(ret < 0, FC.STD(ret, 20), close) ** 2
        output = FC.RANK(FC.TSMAX(inner, 5))
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha11(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = FC.SUM(
            ((close - low) - (high - close)) /
            (high - low) * volume,
            6
        )
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha12(data):
        open = data['open'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = (FC.RANK(open - (FC.SUM(vwap, 10) / 10))) * (-1 * FC.RANK(FC.ABS(close - vwap)))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha13(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        output = ((high * low) ** 0.5) - vwap
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha14(data):
        close = data['close'].reset_index(drop=True)
        output = close - FC.DELAY(close, 5)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha15(data):
        open = data['open'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = open / FC.DELAY(close, 1) - 1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha16(data):
        volume = data['volume'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        output = (-1 * FC.TSMAX(
            FC.RANK(FC.CORR(FC.RANK(volume), FC.RANK(vwap), 5)),
            5
        ))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha17(data):
        vwap = data['vwap'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = FC.RANK((vwap - FC.TSMAX(vwap, 15)) ** FC.DELTA(close, 5))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha18(data):
        close = data['close'].reset_index(drop=True)
        output = close / FC.DELAY(close, 5)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha19(data):
        close = data['close'].reset_index(drop=True)
        delay5 = FC.DELAY(close, 5)
        cond1 = close < delay5
        cond2 = close == delay5
        val1 = (close - delay5) / delay5
        val2 = 0
        val3 = (close - delay5) / close
        output = np.where(cond1, val1, np.where(cond2, val2, val3))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha20(data):
        close = data['close'].reset_index(drop=True)
        output = (close - FC.DELAY(close, 6)) / FC.DELAY(close, 6) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha21(data):
        close = data['close'].reset_index(drop=True)
        output = FC.REGBETA(FC.MEAN(close, 6), FC.SEQUENCE(6), 6)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha22(data):
        close = data['close'].reset_index(drop=True)
        raw = (close - FC.MEAN(close, 6)) / FC.MEAN(close, 6)
        output = FC.SMA(raw - FC.DELAY(raw, 3), 12, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha23(data):
        close = data['close'].reset_index(drop=True)
        cond1 = close > FC.DELAY(close, 1)
        cond2 = close <= FC.DELAY(close, 1)
        vol1 = np.where(cond1, FC.STD(close, 20), 0.0)
        vol2 = np.where(cond2, FC.STD(close, 20), 0.0)
        sma1 = FC.SMA(vol1, 20, 1)
        sma2 = FC.SMA(vol2, 20, 1)
        output = sma1 / (sma1 + sma2) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha24(data):
        close = data['close'].reset_index(drop=True)
        output = FC.SMA(close - FC.DELAY(close, 5), 5, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha25(data):
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        ret = data['ret'].reset_index(drop=True)
        decay = FC.DECAYLINEAR(volume / FC.MEAN(volume, 20), 9)
        output = -1 * FC.RANK(FC.DELTA(close, 7) * (1 - FC.RANK(decay))) * \
            (1 + FC.RANK(FC.SUM(ret, 250)))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha26(data):
        close = data['close'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        output = ((FC.SUM(close, 7) / 7 - close) +
                FC.CORR(vwap, FC.DELAY(close, 5), 230))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha27(data):
        close = data['close'].reset_index(drop=True)
        part = (close - FC.DELAY(close, 3)) / FC.DELAY(close, 3) * 100 + \
            (close - FC.DELAY(close, 6)) / FC.DELAY(close, 6) * 100
        output = FC.WMA(part, 12)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod   
    def alpha28(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        a = (close - FC.TSMIN(low, 9)) / (FC.TSMAX(high, 9) - FC.TSMIN(low, 9)) * 100
        b = (close - FC.TSMIN(low, 9)) / (FC.TSMAX(high, 9) - FC.TSMIN(low, 9)) * 100
        output = 3 * FC.SMA(a, 3, 1) - 2 * FC.SMA(FC.SMA(b, 3, 1), 3, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha29(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (close - FC.DELAY(close, 6)) / FC.DELAY(close, 6) * volume
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha30(data):
        close = data['close'].reset_index(drop=True)
        ret = close / FC.DELAY(close, 1) - 1
        hml_smb_mke = np.column_stack((data['hml'], data['smb'], data['mkt']))
        resid = FC.REGRESI(ret, hml_smb_mke, 60)
        output = FC.WMA(resid ** 2, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha31(data):
        close = data['close'].reset_index(drop=True)
        output = (close - FC.MEAN(close, 12)) / FC.MEAN(close, 12) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha32(data):
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * FC.SUM(FC.RANK(FC.CORR(FC.RANK(high), FC.RANK(volume), 3)), 3)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha33(data):
        low = data['low'].reset_index(drop=True)
        ret = data['ret'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        part1 = -1 * FC.TSMIN(low, 5) + FC.DELAY(FC.TSMIN(low, 5), 5)
        part2 = FC.RANK((FC.SUM(ret, 240) - FC.SUM(ret, 20)) / 220)
        output = part1 * part2 * FC.TSRANK(volume, 5)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha34(data):
        close = data['close'].reset_index(drop=True)
        output = FC.MEAN(close, 12) / close
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha35(data):
        open = data['open'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.DELTA(open, 1), 15))
        corr = FC.CORR(volume, open * 0.65 + close * 0.35, 17)
        rank2 = FC.RANK(FC.DECAYLINEAR(corr, 7))
        output = np.minimum(rank1, rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha36(data):
        volume = data['volume'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        output = FC.RANK(FC.SUM(FC.CORR(FC.RANK(volume), FC.RANK(vwap), 6), 2))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha37(data):
        open = data['open'].reset_index(drop=True)
        ret = data['ret'].reset_index(drop=True)
        output = -1 * FC.RANK(
            (FC.SUM(open, 5) * FC.SUM(ret, 5)) -
            FC.DELAY(FC.SUM(open, 5) * FC.SUM(ret, 5), 10)
        )
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha38(data):
        high = data['high'].reset_index(drop=True)
        output = np.where(FC.SUM(high, 20) / 20 < high, -1 * FC.DELTA(high, 2), 0)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod  
    def alpha39(data):
        close = data['close'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.DELTA(close, 2), 8))
        corr = FC.CORR(vwap * 0.3 + open * 0.7,
                    FC.SUM(FC.MEAN(volume, 180), 37), 14)
        rank2 = FC.RANK(FC.DECAYLINEAR(corr, 12))
        output = (rank1 - rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha40(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        cond = close > FC.DELAY(close, 1)
        output = FC.SUM(np.where(cond, volume, 0), 26) / \
            FC.SUM(np.where(~cond, volume, 0), 26) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha41(data):
        vwap = data['vwap'].reset_index(drop=True)
        output = FC.RANK(FC.TSMAX(FC.DELTA(vwap, 3), 5)) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha42(data):
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (-1 * FC.RANK(FC.STD(high, 10))) * FC.CORR(high, volume, 10)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha43(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        cond1 = close > FC.DELAY(close, 1)
        cond2 = close < FC.DELAY(close, 1)
        val = np.where(cond1, volume, np.where(cond2, -volume, 0))
        output = FC.SUM(val, 6)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha44(data):
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        corr1 = FC.CORR(low, FC.MEAN(volume, 10), 7)
        term1 = FC.TSRANK(FC.DECAYLINEAR(corr1, 6), 4)
        term2 = FC.TSRANK(FC.DECAYLINEAR(FC.DELTA(vwap, 3), 10), 15)
        output = term1 + term2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha45(data):
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DELTA((close * 0.6 + open * 0.4), 1))
        rank2 = FC.RANK(FC.CORR(vwap, FC.MEAN(volume, 150), 15))
        output = rank1 * rank2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha46(data):
        close = data['close'].reset_index(drop=True)
        output = (FC.MEAN(close, 3) + FC.MEAN(close, 6) +
                FC.MEAN(close, 12) + FC.MEAN(close, 24)) / (4 * close)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod   
    def alpha47(data):
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = FC.SMA((FC.TSMAX(high, 6) - close) /
                    (FC.TSMAX(high, 6) - FC.TSMIN(low, 6)) * 100, 9, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha48(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        signs = FC.SIGN(close - FC.DELAY(close, 1)) + \
                FC.SIGN(FC.DELAY(close, 1) - FC.DELAY(close, 2)) + \
                FC.SIGN(FC.DELAY(close, 2) - FC.DELAY(close, 3))
        output = -1 * (FC.RANK(signs) * FC.SUM(volume, 5)) / FC.SUM(volume, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha49(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        cond_up = (high + low) >= (FC.DELAY(high, 1) + FC.DELAY(low, 1))
        cond_down = (high + low) <= (FC.DELAY(high, 1) + FC.DELAY(low, 1))
        diff = np.maximum(FC.ABS(high - FC.DELAY(high, 1)),
                        FC.ABS(low - FC.DELAY(low, 1)))
        up = FC.SUM(np.where(cond_up, 0, diff), 12)
        down = FC.SUM(np.where(cond_down, 0, diff), 12)
        output = up / (up + down)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha50(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        cond_up = (high + low) >= (FC.DELAY(high, 1) + FC.DELAY(low, 1))
        cond_down = (high + low) <= (FC.DELAY(high, 1) + FC.DELAY(low, 1))
        diff = np.maximum(FC.ABS(high - FC.DELAY(high, 1)),
                        FC.ABS(low - FC.DELAY(low, 1)))
        up = FC.SUM(np.where(cond_up, 0, diff), 12)
        down = FC.SUM(np.where(cond_down, 0, diff), 12)
        output = down / (down + up) - up / (up + down)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha51(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        cond_down = (high + low) <= (FC.DELAY(high, 1) + FC.DELAY(low, 1))
        cond_up = (high + low) >= (FC.DELAY(high, 1) + FC.DELAY(low, 1))
        diff = np.maximum(FC.ABS(high - FC.DELAY(high, 1)),
                        FC.ABS(low - FC.DELAY(low, 1)))
        down = FC.SUM(np.where(cond_down, 0, diff), 12)
        up = FC.SUM(np.where(cond_up, 0, diff), 12)
        output = down / (down + up)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha52(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        mid = (high + low + close) / 3
        diff1 = np.maximum(0, high - FC.DELAY(mid, 1))
        diff2 = np.maximum(0, FC.DELAY(mid, 1) - low)
        output = FC.SUM(diff1, 26) / FC.SUM(diff2, 26) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha53(data):
        close = data['close'].reset_index(drop=True)
        output = FC.COUNT(close > FC.DELAY(close, 1), 12) / 12 * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha54(data):
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        output = -1 * FC.RANK(
            FC.STD(FC.ABS(close - open), 10) +
            (close - open) +
            FC.CORR(close, open, 10)
        )
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha55(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        a = FC.ABS(high - FC.DELAY(close, 1))
        b = FC.ABS(low - FC.DELAY(close, 1))
        c = FC.ABS(high - FC.DELAY(low, 1))
        d = FC.ABS(FC.DELAY(close, 1) - FC.DELAY(open, 1))
        cond1 = (a > b) & (a > c)
        cond2 = (b > c) & (b > a)
        denom = np.where(cond1, a + b/2 + d/4,
                        np.where(cond2, b + a/2 + d/4,
                                c + d/4))
        num = close - FC.DELAY(close, 1) + (close - open)/2 + \
            FC.DELAY(close, 1) - FC.DELAY(open, 1)
        output = FC.SUM(16 * num / denom * np.maximum(a, b), 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha56(data):
        open = data['open'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(open - FC.TSMIN(open, 12))
        corr = FC.CORR(FC.SUM((high + low) / 2, 19),
                    FC.SUM(FC.MEAN(volume, 40), 19), 13)
        rank2 = FC.RANK(FC.RANK(corr) ** 5)
        output = (rank1 < rank2).astype(float)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha57(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        output = FC.SMA((close - FC.TSMIN(low, 9)) /
                    (FC.TSMAX(high, 9) - FC.TSMIN(low, 9)) * 100, 3, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha58(data):
        close = data['close'].reset_index(drop=True)
        output = FC.COUNT(close > FC.DELAY(close, 1), 20) / 20 * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha59(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        cond = close == FC.DELAY(close, 1)
        val_false = close - np.where(
            close > FC.DELAY(close, 1),
            np.minimum(low, FC.DELAY(close, 1)),
            np.maximum(high, FC.DELAY(close, 1))
        )
        val = np.where(cond, 0, val_false)
        output = FC.SUM(val, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha60(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = FC.SUM(
            ((close - low) - (high - close)) /
            (high - low) * volume,
            20
        )
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha61(data):
        vwap = data['vwap'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.DELTA(vwap, 1), 12))
        corr = FC.CORR(low, FC.MEAN(volume, 80), 8)
        rank2 = FC.RANK(FC.DECAYLINEAR(FC.RANK(corr), 17))
        output = np.maximum(rank1, rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha62(data):
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * FC.CORR(high, FC.RANK(volume), 5)
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha63(data):
        close = data['close'].reset_index(drop=True)
        diff = close - FC.DELAY(close, 1)
        output = FC.SMA(np.maximum(diff, 0), 6, 1) / FC.SMA(FC.ABS(diff), 6, 1) * 100
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha64(data):
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.CORR(FC.RANK(vwap), FC.RANK(volume), 4), 4))
        corr = FC.CORR(FC.RANK(close), FC.RANK(FC.MEAN(volume, 60)), 4)
        rank2 = FC.RANK(FC.DECAYLINEAR(np.maximum(corr, 13), 14))
        output = np.maximum(rank1, rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha65(data):
        close = data['close'].reset_index(drop=True)
        output = FC.MEAN(close, 6) / close
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha66(data):
        close = data['close'].reset_index(drop=True)
        output = (close - FC.MEAN(close, 6)) / FC.MEAN(close, 6) * 100
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha67(data):
        close = data['close'].reset_index(drop=True)
        diff = close - FC.DELAY(close, 1)
        output = FC.SMA(np.maximum(diff, 0), 24, 1) / FC.SMA(FC.ABS(diff), 24, 1) * 100
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha68(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = FC.SMA(
            ((high + low) / 2 - (FC.DELAY(high, 1) + FC.DELAY(low, 1)) / 2) *
            (high - low) / volume, 15, 2
        )
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha69(data):
        open = data['open'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        DTM = np.where(open < FC.DELAY(open, 1), 0,
                    np.maximum(high - open,
                                open - FC.DELAY(open, 1)))
        DBM = np.where(open > FC.DELAY(open, 1), 0,
                    np.maximum(open - low,
                                open - FC.DELAY(open, 1)))
        sum_dtm = FC.SUM(DTM, 20)
        sum_dbm = FC.SUM(DBM, 20)
        cond1 = sum_dtm > sum_dbm
        cond2 = sum_dtm == sum_dbm
        output = np.where(cond1, (sum_dtm - sum_dbm) / sum_dtm,
                        np.where(cond2, 0.0, (sum_dtm - sum_dbm) / sum_dbm))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha70(data):
        amount = data['amount'].reset_index(drop=True)
        output = FC.STD(amount, 6)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha71(data):
        close = data['close'].reset_index(drop=True)
        output = (close - FC.MEAN(close, 24)) / FC.MEAN(close, 24) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha72(data):
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = FC.SMA((FC.TSMAX(high, 6) - close) /
                    (FC.TSMAX(high, 6) - FC.TSMIN(low, 6)) * 100, 15, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod   
    def alpha73(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        rank1 = FC.TSRANK(FC.DECAYLINEAR(FC.DECAYLINEAR(FC.CORR(close, volume, 10), 16), 4), 5)
        rank2 = FC.RANK(FC.DECAYLINEAR(FC.CORR(vwap, FC.MEAN(volume, 30), 4), 3))
        output = (rank1 - rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha74(data):
        low = data['low'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        part1 = FC.RANK(FC.CORR(FC.SUM(low * 0.35 + vwap * 0.65, 20),
                                FC.SUM(FC.MEAN(volume, 40), 20), 7))
        part2 = FC.RANK(FC.CORR(FC.RANK(vwap), FC.RANK(volume), 6))
        output = part1 + part2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha75(data):
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        benchmarkindexclose = data['benchmarkindexclose'].reset_index(drop=True)
        benchmarkindexopen = data['benchmarkindexopen'].reset_index(drop=True)
        cond = (close > open) & (benchmarkindexclose < benchmarkindexopen)
        cond_bench = benchmarkindexclose < benchmarkindexopen
        output = FC.COUNT(cond, 50) / FC.COUNT(cond_bench, 50)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha76(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        ret = close / FC.DELAY(close, 1) - 1
        ratio = FC.ABS(ret) / volume
        output = FC.STD(ratio, 20) / FC.MEAN(ratio, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha77(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(((high + low) / 2 + high - (vwap + high)), 20))
        corr = FC.CORR((high + low) / 2, FC.MEAN(volume, 40), 3)
        rank2 = FC.RANK(FC.DECAYLINEAR(corr, 6))
        output = np.minimum(rank1, rank2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha78(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        avg = (high + low + close) / 3
        ma12 = FC.MEAN(avg, 12)
        numerator = avg - ma12
        denominator = 0.015 * FC.MEAN(FC.ABS(close - FC.MEAN(avg, 12)), 12)
        output = numerator / denominator
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha79(data):
        close = data['close'].reset_index(drop=True)
        diff = close - FC.DELAY(close, 1)
        output = FC.SMA(np.maximum(diff, 0), 12, 1) / FC.SMA(FC.ABS(diff), 12, 1) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha80(data):
        volume = data['volume'].reset_index(drop=True)
        output = (volume - FC.DELAY(volume, 5)) / FC.DELAY(volume, 5) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha81(data):
        volume = data['volume'].reset_index(drop=True)
        output = FC.SMA(volume, 21, 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha82(data):
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = FC.SMA((FC.TSMAX(high, 6) - close) /
                    (FC.TSMAX(high, 6) - FC.TSMIN(low, 6)) * 100, 20, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha83(data):
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * FC.RANK(FC.COVIANCE(FC.RANK(high), FC.RANK(volume), 5))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha84(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        cond1 = close > FC.DELAY(close, 1)
        cond2 = close < FC.DELAY(close, 1)
        val = np.where(cond1, volume, np.where(cond2, -volume, 0))
        output = FC.SUM(val, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha85(data):
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = FC.TSRANK(volume / FC.MEAN(volume, 20), 20) * \
            FC.TSRANK(-1 * FC.DELTA(close, 7), 8)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha86(data):
        close = data['close'].reset_index(drop=True)
        slope = ((FC.DELAY(close, 20) - FC.DELAY(close, 10)) / 10) - \
                ((FC.DELAY(close, 10) - close) / 10)
        cond1 = slope > 0.25
        cond2 = slope < 0
        output = np.where(cond1, -1.0, np.where(cond2, 1.0, -1.0 * (close - FC.DELAY(close, 1))))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha87(data):
        vwap = data['vwap'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.DELTA(vwap, 4), 7))
        val = (low - vwap) / (open - (high + low) / 2)
        rank2 = FC.TSRANK(FC.DECAYLINEAR(val, 11), 7)
        output = (rank1 + rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod   
    def alpha88(data):
        close = data['close'].reset_index(drop=True)
        output = (close - FC.DELAY(close, 20)) / FC.DELAY(close, 20) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha89(data):
        close = data['close'].reset_index(drop=True)
        sma13 = FC.SMA(close, 13, 2)
        sma27 = FC.SMA(close, 27, 2)
        output = 2 * (sma13 - sma27 - FC.SMA(sma13 - sma27, 10, 2))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha90(data):
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = FC.RANK(FC.CORR(FC.RANK(vwap), FC.RANK(volume), 5)) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha91(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        rank1 = FC.RANK(close - FC.TSMAX(close, 5))
        rank2 = FC.RANK(FC.CORR(FC.MEAN(volume, 40), low, 5))
        output = rank1 * rank2 * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha92(data):
        close = data['close'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(
            FC.DELTA(close * 0.35 + vwap * 0.65, 2), 3))
        corr = FC.CORR(FC.MEAN(volume, 180), close, 13)
        term2 = FC.TSRANK(FC.DECAYLINEAR(FC.ABS(corr), 5), 15)
        output = -1 * np.maximum(rank1, term2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha93(data):
        open = data['open'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        cond = open >= FC.DELAY(open, 1)
        val = np.where(cond, 0,
                    np.maximum(open - low, open - FC.DELAY(open, 1)))
        output = FC.SUM(val, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha94(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        cond1 = close > FC.DELAY(close, 1)
        cond2 = close < FC.DELAY(close, 1)
        val = np.where(cond1, volume, np.where(cond2, -volume, 0))
        output = FC.SUM(val, 30)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha95(data):
        amount = data['amount'].reset_index(drop=True)
        output = FC.STD(amount, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha96(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        raw = (close - FC.TSMIN(low, 9)) / \
            (FC.TSMAX(high, 9) - FC.TSMIN(low, 9)) * 100
        output = FC.SMA(FC.SMA(raw, 3, 1), 3, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha97(data):
        volume = data['volume'].reset_index(drop=True)
        output = FC.STD(volume, 10)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha98(data):
        close = data['close'].reset_index(drop=True)
        ma100 = FC.SUM(close, 100) / 100
        ratio = FC.DELTA(ma100, 100) / FC.DELAY(close, 100)
        cond = (ratio < 0.05) | (ratio == 0.05)
        output = np.where(cond, -1 * (close - FC.TSMIN(close, 100)),
                        -1 * FC.DELTA(close, 3))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha99(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * FC.RANK(FC.COVIANCE(FC.RANK(close), FC.RANK(volume), 5))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha100(data):
        volume = data['volume'].reset_index(drop=True)
        output = FC.STD(volume, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha101(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        rank1 = FC.RANK(FC.CORR(close, FC.SUM(FC.MEAN(volume, 30), 37), 15))
        rank2 = FC.RANK(FC.CORR(FC.RANK(high * 0.1 + vwap * 0.9),
                                FC.RANK(volume), 11))
        output = (rank1 < rank2).astype(float) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha102(data):
        volume = data['volume'].reset_index(drop=True)
        diff = volume - FC.DELAY(volume, 1)
        output = FC.SMA(np.maximum(diff, 0), 6, 1) / FC.SMA(FC.ABS(diff), 6, 1) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha103(data):
        low = data['low'].reset_index(drop=True)
        output = (20 - FC.LOWDAY(low, 20)) / 20 * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha104(data):
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = -1 * FC.DELTA(FC.CORR(high, volume, 5), 5) * FC.RANK(FC.STD(close, 20))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha105(data):
        open = data['open'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * FC.CORR(FC.RANK(open), FC.RANK(volume), 10)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha106(data):
        close = data['close'].reset_index(drop=True)
        output = close - FC.DELAY(close, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha107(data):
        open = data['open'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = -1 * FC.RANK(open - FC.DELAY(high, 1)) * \
            FC.RANK(open - FC.DELAY(close, 1)) * \
            FC.RANK(open - FC.DELAY(low, 1))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha108(data):
        high = data['high'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(high - FC.TSMIN(high, 2))
        rank2 = FC.RANK(FC.CORR(vwap, FC.MEAN(volume, 120), 6))
        output = (rank1 ** rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha109(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        diff = high - low
        sma1 = FC.SMA(diff, 10, 2)
        output = sma1 / FC.SMA(sma1, 10, 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha110(data):
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        up = np.maximum(0, high - FC.DELAY(close, 1))
        dn = np.maximum(0, FC.DELAY(close, 1) - low)
        output = FC.SUM(up, 20) / FC.SUM(dn, 20) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha111(data):
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        vol = volume
        raw = vol * ((close - low) - (high - close)) / (high - low)
        output = FC.SMA(raw, 11, 2) - FC.SMA(raw, 4, 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha112(data):
        close = data['close'].reset_index(drop=True)
        delta = close - FC.DELAY(close, 1)
        pos = FC.SUM(np.where(delta > 0, delta, 0), 12)
        neg_abs = FC.SUM(np.where(delta < 0, FC.ABS(delta), 0), 12)
        output = (pos - neg_abs) / (pos + neg_abs) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha113(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * (FC.RANK(FC.SUM(FC.DELAY(close, 5), 20) / 20) *
                    FC.CORR(close, volume, 2) *
                    FC.RANK(FC.CORR(FC.SUM(close, 5), FC.SUM(close, 20), 2)))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha114(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        spread = (high - low) / (FC.SUM(close, 5) / 5)
        rank1 = FC.RANK(FC.DELAY(spread, 2))
        rank2 = FC.RANK(volume.rank())   # 注意：此处 volume.rank() 是 pandas Series 方法，保留原意
        output = rank1 * rank2 / (spread / (vwap - close))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha115(data):
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        rank1 = FC.RANK(FC.CORR(high * 0.9 + close * 0.1,
                                FC.MEAN(volume, 30), 10))
        rank2 = FC.RANK(FC.CORR(FC.TSRANK((high + low) / 2, 4),
                                FC.TSRANK(volume, 10), 7))
        output = rank1 ** rank2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha116(data):
        close = data['close'].reset_index(drop=True)
        output = FC.REGBETA(close, FC.SEQUENCE(20), 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha117(data):
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        ret = data['ret'].reset_index(drop=True)
        vol_rank = FC.TSRANK(volume, 32)
        price_rank = FC.TSRANK(close + high - low, 16)
        ret_rank = FC.TSRANK(ret, 32)
        output = vol_rank * (1 - price_rank) * (1 - ret_rank)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha118(data):
        high = data['high'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = FC.SUM(high - open, 20) / FC.SUM(open - low, 20) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha119(data):
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        corr1 = FC.CORR(vwap, FC.SUM(FC.MEAN(volume, 5), 26), 5)
        rank1 = FC.RANK(FC.DECAYLINEAR(corr1, 7))
        corr2 = FC.CORR(FC.RANK(open), FC.RANK(FC.MEAN(volume, 15)), 21)
        rank2 = FC.RANK(FC.DECAYLINEAR(FC.TSRANK(np.minimum(corr2, 9), 7), 8))
        output = rank1 - rank2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha120(data):
        vwap = data['vwap'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = FC.RANK(vwap - close) / FC.RANK(vwap + close)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha121(data):
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(vwap - FC.TSMIN(vwap, 12))
        ts = FC.TSRANK(FC.CORR(FC.TSRANK(vwap, 20),
                            FC.TSRANK(FC.MEAN(volume, 60), 2), 18), 3)
        output = (rank1 ** ts) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha122(data):
        close = data['close'].reset_index(drop=True)
        log_close = FC.LOG(close)
        sma1 = FC.SMA(FC.SMA(FC.SMA(log_close, 13, 2), 13, 2), 13, 2)
        sma2 = FC.DELAY(sma1, 1)
        output = (sma1 - sma2) / sma2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha123(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.CORR(FC.SUM((high + low) / 2, 20),
                                FC.SUM(FC.MEAN(volume, 60), 20), 9))
        rank2 = FC.RANK(FC.CORR(low, volume, 6))
        output = (rank1 < rank2).astype(float) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha124(data):
        close = data['close'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        output = (close - vwap) / FC.DECAYLINEAR(FC.RANK(FC.TSMAX(close, 30)), 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha125(data):
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.CORR(vwap, FC.MEAN(volume, 80), 17), 20))
        rank2 = FC.RANK(FC.DECAYLINEAR(FC.DELTA(close * 0.5 + vwap * 0.5, 3), 16))
        output = rank1 / rank2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha126(data):
        close = data['close'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = (close + high + low) / 3
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha127(data):
        close = data['close'].reset_index(drop=True)
        diff = 100 * (close - FC.TSMAX(close, 12)) / FC.TSMAX(close, 12)
        output = np.sqrt(FC.MEAN(diff ** 2, 1))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha128(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        avg = (high + low + close) / 3
        up = np.where(avg > FC.DELAY(avg, 1), avg * volume, 0)
        down = np.where(avg < FC.DELAY(avg, 1), avg * volume, 0)
        sum_up = FC.SUM(up, 14)
        sum_down = FC.SUM(down, 14)
        output = 100 - (100 / (1 + sum_up / sum_down))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha129(data):
        close = data['close'].reset_index(drop=True)
        output = FC.SUM(np.where(close - FC.DELAY(close, 1) < 0,
                            FC.ABS(close - FC.DELAY(close, 1)), 0), 12)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha130(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.CORR((high + low) / 2,
                                                FC.MEAN(volume, 40), 9), 10))
        rank2 = FC.RANK(FC.DECAYLINEAR(FC.CORR(FC.RANK(vwap), FC.RANK(volume), 7), 3))
        output = rank1 / rank2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha131(data):
        vwap = data['vwap'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DELTA(vwap, 1))
        ts = FC.TSRANK(FC.CORR(close, FC.MEAN(volume, 50), 18), 18)
        output = rank1 ** ts
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha132(data):
        amount = data['amount'].reset_index(drop=True)
        output = FC.MEAN(amount, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha133(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = ((20 - FC.HIGHDAY(high, 20)) / 20) * 100 - ((20 - FC.LOWDAY(low, 20)) / 20) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha134(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (close - FC.DELAY(close, 12)) / FC.DELAY(close, 12) * volume
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha135(data):
        close = data['close'].reset_index(drop=True)
        output = FC.SMA(FC.DELAY(close / FC.DELAY(close, 20), 1), 20, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha136(data):
        ret = data['ret'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * FC.RANK(FC.DELTA(ret, 3)) * FC.CORR(open, volume, 10)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha137(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        a = FC.ABS(high - FC.DELAY(close, 1))
        b = FC.ABS(low - FC.DELAY(close, 1))
        c = FC.ABS(high - FC.DELAY(low, 1))
        d = FC.ABS(FC.DELAY(close, 1) - FC.DELAY(open, 1))
        cond1 = (a > b) & (a > c)
        cond2 = (b > c) & (b > a)
        denom = np.where(cond1, a + b/2 + d/4,
                        np.where(cond2, b + a/2 + d/4,
                                c + d/4))
        num = close - FC.DELAY(close, 1) + (close - open)/2 + \
            FC.DELAY(close, 1) - FC.DELAY(open, 1)
        output = 16 * num / denom * np.maximum(a, b)
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha138(data):
        low = data['low'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.DELTA(low * 0.7 + vwap * 0.3, 3), 20))
        corr = FC.TSRANK(FC.CORR(FC.TSRANK(low, 8), FC.TSRANK(FC.MEAN(volume, 60), 17), 5), 19)
        rank2 = FC.TSRANK(FC.DECAYLINEAR(corr, 16), 7)
        output = (rank1 - rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha139(data):
        open = data['open'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = -1 * FC.CORR(open, volume, 10)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha140(data):
        open = data['open'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR((FC.RANK(open) + FC.RANK(low) -
                                        FC.RANK(high) - FC.RANK(close)), 8))
        corr = FC.CORR(FC.TSRANK(close, 8), FC.TSRANK(FC.MEAN(volume, 60), 20), 8)
        rank2 = FC.TSRANK(FC.DECAYLINEAR(corr, 7), 3)
        output = np.minimum(rank1, rank2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha141(data):
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = FC.RANK(FC.CORR(FC.RANK(high), FC.RANK(FC.MEAN(volume, 15)), 9)) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha142(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (-1 * FC.RANK(FC.TSRANK(close, 10)) *
                FC.RANK(FC.DELTA(FC.DELTA(close, 1), 1)) *
                FC.RANK(FC.TSRANK(volume / FC.MEAN(volume, 20), 5)))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha143(data, self_series=None):
        close = data['close'].reset_index(drop=True)
        if self_series is None:
            self_series = pd.Series(np.nan, index=data.index)
            self_series.iloc[0] = 0.0
        for i in range(1, len(data)):
            if close.iloc[i] > close.iloc[i-1]:
                self_series.iloc[i] = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * self_series.iloc[i-1]
            else:
                self_series.iloc[i] = self_series.iloc[i-1]
        output = self_series
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha144(data):
        close = data['close'].reset_index(drop=True)
        amount = data['amount'].reset_index(drop=True)
        cond = close < FC.DELAY(close, 1)
        num = FC.ABS(close / FC.DELAY(close, 1) - 1) / amount
        den = FC.COUNT(cond, 20)
        output = FC.SUMIF(num, 20, cond)/den
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha145(data):
        volume = data['volume'].reset_index(drop=True)
        output = (FC.MEAN(volume, 9) - FC.MEAN(volume, 26)) / FC.MEAN(volume, 12) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha146(data):
        close = data['close'].reset_index(drop=True)
        cls = (close - FC.DELAY(close, 1)) / FC.DELAY(close, 1)
        sma = FC.SMA((close - FC.DELAY(close, 1)) / FC.DELAY(close, 1), 61, 2)
        comp1 = FC.MEAN(cls - sma, 20)
        comp2 = cls - sma
        comp3 = FC.SMA((cls - sma) ** 2, 60, 2)
        output = comp1 * comp2 / comp3
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha147(data):
        close = data['close'].reset_index(drop=True)
        output = FC.REGBETA(FC.MEAN(close, 12), FC.SEQUENCE(12), 12)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha148(data):
        open = data['open'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK(FC.CORR(open, FC.SUM(FC.MEAN(volume, 60), 9), 6))
        rank2 = FC.RANK(open - FC.TSMIN(open, 14))
        output = (rank1 < rank2).astype(float) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha149(data):
        close = data['close'].reset_index(drop=True)
        benchmarkindexclose = data['benchmarkindexclose'].reset_index(drop=True)
        filter1 = FC.FILTER((close / FC.DELAY(close, 1) - 1), 
                            benchmarkindexclose < FC.DELAY(benchmarkindexclose, 1))
        filter2 = FC.FILTER((benchmarkindexclose / FC.DELAY(benchmarkindexclose, 1) - 1), 
                            benchmarkindexclose < FC.DELAY(benchmarkindexclose, 1))
        output = FC.REGBETA(filter1, filter2, 252)
        return np.nan_to_num(np.array(output), copy=False)
    
    @staticmethod
    def alpha150(data):
        close = data['close'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (close + high + low) / 3 * volume
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha151(data):
        close = data['close'].reset_index(drop=True)
        output = FC.SMA(close - FC.DELAY(close, 20), 20, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha152(data):
        close = data['close'].reset_index(drop=True)
        ret = close / FC.DELAY(close, 9)
        sma1 = FC.SMA(FC.DELAY(ret, 1), 9, 1)
        ma12 = FC.MEAN(FC.DELAY(sma1, 1), 12)
        ma26 = FC.MEAN(FC.DELAY(sma1, 1), 26)
        output = FC.SMA(ma12 - ma26, 9, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha153(data):
        close = data['close'].reset_index(drop=True)
        output = (FC.MEAN(close, 3) + FC.MEAN(close, 6) +
                FC.MEAN(close, 12) + FC.MEAN(close, 24)) / 4
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha154(data):
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (vwap - FC.TSMIN(vwap, 16)) < \
            FC.CORR(vwap, FC.MEAN(volume, 180), 18)
        return np.array(output, dtype=float)

    @staticmethod
    def alpha155(data):
        volume = data['volume'].reset_index(drop=True)
        sma13 = FC.SMA(volume, 13, 2)
        sma27 = FC.SMA(volume, 27, 2)
        output = sma13 - sma27 - FC.SMA(sma13 - sma27, 10, 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha156(data):
        vwap = data['vwap'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        rank1 = FC.RANK(FC.DECAYLINEAR(FC.DELTA(vwap, 5), 3))
        rank2 = FC.RANK(FC.DECAYLINEAR(
            -1 * FC.DELTA(open * 0.15 + low * 0.85, 2) /
            (open * 0.15 + low * 0.85), 3))
        output = np.maximum(rank1, rank2) * -1
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha157(data):
        close = data['close'].reset_index(drop=True)
        ret = data['ret'].reset_index(drop=True)
        term1 = FC.PROD(FC.RANK(FC.RANK(FC.LOG(FC.SUM(
            FC.TSMIN(FC.RANK(FC.RANK(-1 * FC.RANK(FC.DELTA(close - 1, 5)))), 2), 1)))), 5)
        term2 = FC.TSRANK(FC.DELAY(-1 * ret, 6), 5)
        output = np.minimum(term1, 5) + term2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha158(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = ((high - FC.SMA(close, 15, 2)) -
                (low - FC.SMA(close, 15, 2))) / close
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha159(data):
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        min_low = np.minimum(low, FC.DELAY(close, 1))
        max_high = np.maximum(high, FC.DELAY(close, 1))
        diff = max_high - min_low
        term6 = (close - FC.SUM(min_low, 6)) / FC.SUM(diff, 6) * 12 * 24
        term12 = (close - FC.SUM(min_low, 12)) / FC.SUM(diff, 12) * 6 * 24
        term24 = (close - FC.SUM(min_low, 24)) / FC.SUM(diff, 24) * 6 * 24
        output = (term6 + term12 + term24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha160(data):
        close = data['close'].reset_index(drop=True)
        cond = close <= FC.DELAY(close, 1)
        output = FC.SMA(np.where(cond, FC.STD(close, 20), 0), 20, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha161(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        tr = np.maximum(np.maximum(high - low,
                                FC.ABS(high - FC.DELAY(close, 1))),
                        FC.ABS(low - FC.DELAY(close, 1)))
        output = FC.MEAN(tr, 12)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha162(data):
        close = data['close'].reset_index(drop=True)
        diff = close - FC.DELAY(close, 1)
        rsi = FC.SMA(np.maximum(diff, 0), 12, 1) / FC.SMA(FC.ABS(diff), 12, 1) * 100
        min_rsi = FC.TSMIN(rsi, 12)
        max_rsi = FC.TSMAX(rsi, 12)
        output = (rsi - min_rsi) / (max_rsi - min_rsi)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha163(data):
        ret = data['ret'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = FC.RANK((-1 * ret * FC.MEAN(volume, 20) * vwap) * (high - close))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha164(data):
        close = data['close'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        inv = np.where(close > FC.DELAY(close, 1),
                    1 / (close - FC.DELAY(close, 1)), 1)
        min_inv = FC.TSMIN(inv, 12)
        output = FC.SMA((inv - min_inv) / (high - low) * 100, 13, 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha165(data):
        close = data['close'].reset_index(drop=True)
        cum = FC.SUMAC(close - FC.MEAN(close, 48), 1)
        output = (np.max(cum) - np.min(cum)) / FC.STD(close, 48)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha166(data):
        close = data['close'].reset_index(drop=True)
        ret = close / FC.DELAY(close, 1) - 1
        mean_ret = FC.MEAN(ret, 20)
        numerator = FC.SUM(ret - mean_ret, 20)
        denominator = FC.SUM((ret) ** 2, 20)
        output = -20 * (19 ** 1.5) * numerator / ( (19) * (20) * (denominator ** 1.5) )
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha167(data):
        close = data['close'].reset_index(drop=True)
        output = FC.SUM(np.where(close - FC.DELAY(close, 1) > 0,
                            close - FC.DELAY(close, 1), 0), 12)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha168(data):
        volume = data['volume'].reset_index(drop=True)
        output = -1 * volume / FC.MEAN(volume, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha169(data):
        close = data['close'].reset_index(drop=True)
        diff = close - FC.DELAY(close, 1)
        sma = FC.SMA(diff, 9, 1)
        ma12 = FC.MEAN(FC.DELAY(sma, 1), 12)
        ma26 = FC.MEAN(FC.DELAY(sma, 1), 26)
        output = FC.SMA(ma12 - ma26, 10, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha170(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        vwap = data['vwap'].reset_index(drop=True)
        term1 = (FC.RANK(1 / close) * volume) / FC.MEAN(volume, 20)
        term2 = (high * FC.RANK(high - close)) / (FC.SUM(high, 5) / 5)
        term3 = FC.RANK(vwap - FC.DELAY(vwap, 5))
        output = term1 * term2 - term3
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha171(data):
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        output = (-1 * ((low - close) * (open ** 5))) / \
            ((close - high) * (close ** 5))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha172(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        HD = high - FC.DELAY(high, 1)
        LD = FC.DELAY(low, 1) - low
        TR = np.maximum(np.maximum(high - low,
                                FC.ABS(high - FC.DELAY(close, 1))),
                        FC.ABS(low - FC.DELAY(close, 1)))
        up = np.where((LD > 0) & (LD > HD), LD, 0)
        down = np.where((HD > 0) & (HD > LD), HD, 0)
        sum_up = FC.SUM(up, 14)
        sum_down = FC.SUM(down, 14)
        sum_tr = FC.SUM(TR, 14)
        rsi_up = sum_up * 100 / sum_tr
        rsi_down = sum_down * 100 / sum_tr
        raw = FC.ABS(rsi_up - rsi_down) / (rsi_up + rsi_down) * 100
        output = FC.MEAN(raw, 6)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha173(data):
        close = data['close'].reset_index(drop=True)
        sma13 = FC.SMA(close, 13, 2)
        log_close = FC.LOG(close)
        output = 3 * sma13 - 2 * FC.SMA(sma13, 13, 2) + \
            FC.SMA(FC.SMA(FC.SMA(log_close, 13, 2), 13, 2), 13, 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha174(data):
        close = data['close'].reset_index(drop=True)
        cond = close > FC.DELAY(close, 1)
        output = FC.SMA(np.where(cond, FC.STD(close, 20), 0), 20, 1)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha175(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        tr = np.maximum(np.maximum(high - low,
                                FC.ABS(high - FC.DELAY(close, 1))),
                        FC.ABS(low - FC.DELAY(close, 1)))
        output = FC.MEAN(tr, 6)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha176(data):
        close = data['close'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        rank1 = FC.RANK((close - FC.TSMIN(low, 12)) /
                        (FC.TSMAX(high, 12) - FC.TSMIN(low, 12)))
        output = FC.CORR(rank1, FC.RANK(volume), 6)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha177(data):
        high = data['high'].reset_index(drop=True)
        output = ((20 - FC.HIGHDAY(high, 20)) / 20) * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha178(data):
        close = data['close'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        output = (close - FC.DELAY(close, 1)) / FC.DELAY(close, 1) * volume
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha179(data):
        vwap = data['vwap'].reset_index(drop=True)
        volume = data['volume'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        output = FC.RANK(FC.CORR(vwap, volume, 4)) * \
            FC.RANK(FC.CORR(FC.RANK(low), FC.RANK(FC.MEAN(volume, 50)), 12))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha180(data):
        volume = data['volume'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        cond = volume > FC.MEAN(volume, 20)
        val1 = -1 * FC.TSRANK(FC.ABS(FC.DELTA(close, 7)), 60) * FC.SIGN(FC.DELTA(close, 7))
        output = np.where(cond, val1, -1 * volume)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha181(data):
        close = data['close'].reset_index(drop=True)
        benchmarkindexclose = data['benchmarkindexclose'].reset_index(drop=True)
        ret = close / FC.DELAY(close, 1) - 1
        bench_ret = benchmarkindexclose / FC.DELAY(benchmarkindexclose, 1) - 1
        diff = ret - FC.MEAN(ret, 20)
        bench_diff = bench_ret - FC.MEAN(bench_ret, 20)
        num = FC.SUM(diff - bench_diff ** 2, 20)
        den = FC.SUM(bench_diff ** 3, 20)
        output = num / den
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha182(data):
        close = data['close'].reset_index(drop=True)
        open = data['open'].reset_index(drop=True)
        benchmarkindexclose = data['benchmarkindexclose'].reset_index(drop=True)
        benchmarkindexopen = data['benchmarkindexopen'].reset_index(drop=True)
        cond1 = (close > open) & (benchmarkindexclose > benchmarkindexopen)
        cond2 = (close < open) & (benchmarkindexclose < benchmarkindexopen)
        output = FC.COUNT(cond1 | cond2, 20) / 20
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha183(data):
        close = data['close'].reset_index(drop=True)
        cum = FC.SUMAC(close - FC.MEAN(close, 24), 24)
        output = (cum.expanding().max() - cum.expanding().min()) / FC.STD(close, 24)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha184(data):
        open = data['open'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = FC.RANK(FC.CORR(FC.DELAY(open - close, 1), close, 200)) + \
            FC.RANK(open - close)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha185(data):
        open = data['open'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = FC.RANK(-1 * (1 - open / close) ** 2)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha186(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        HD = high - FC.DELAY(high, 1)
        LD = FC.DELAY(low, 1) - low
        TR = np.maximum(np.maximum(high - low,
                                FC.ABS(high - FC.DELAY(close, 1))),
                        FC.ABS(low - FC.DELAY(close, 1)))
        up = np.where((LD > 0) & (LD > HD), LD, 0)
        down = np.where((HD > 0) & (HD > LD), HD, 0)
        sum_up = FC.SUM(up, 14)
        sum_down = FC.SUM(down, 14)
        sum_tr = FC.SUM(TR, 14)
        rsi_up = sum_up * 100 / sum_tr
        rsi_down = sum_down * 100 / sum_tr
        raw = FC.ABS(rsi_up - rsi_down) / (rsi_up + rsi_down) * 100
        ma6 = FC.MEAN(raw, 6)
        delay6 = FC.DELAY(ma6, 6)
        output = (ma6 + delay6) / 2
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha187(data):
        open = data['open'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        cond = open <= FC.DELAY(open, 1)
        val = np.where(cond, 0,
                    np.maximum(high - open, open - FC.DELAY(open, 1)))
        output = FC.SUM(val, 20)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha188(data):
        high = data['high'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        diff = high - low
        sma = FC.SMA(diff, 11, 2)
        output = (diff - sma) / sma * 100
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha189(data):
        close = data['close'].reset_index(drop=True)
        output = FC.MEAN(FC.ABS(close - FC.MEAN(close, 6)), 6)
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha190(data):
        close = data['close'].reset_index(drop=True)
        ret = close / FC.DELAY(close, 1) - 1
        bench = (close / FC.DELAY(close, 19)) ** (1 / 20) - 1
        cond1 = ret > bench
        cond2 = ret < bench
        cnt1 = FC.COUNT(cond1, 20) - 1
        cnt2 = FC.COUNT(cond2, 20)
        sum1 = FC.SUMIF((ret - bench) ** 2, 20, cond2)
        sum2 = FC.SUMIF((ret - bench) ** 2, 20, cond1)
        output = FC.LOG(cnt1 * sum1 / (cnt2 * sum2))
        return np.nan_to_num(np.array(output), copy=False)

    @staticmethod
    def alpha191(data):
        volume = data['volume'].reset_index(drop=True)
        low = data['low'].reset_index(drop=True)
        high = data['high'].reset_index(drop=True)
        close = data['close'].reset_index(drop=True)
        output = FC.CORR(FC.MEAN(volume, 20), low, 5) + (high + low) / 2 - close
        return np.nan_to_num(np.array(output), copy=False)

    # ---------- 批量计算接口 ----------

    @classmethod
    def compute_factor_for_universe(
        cls,
        data: pd.DataFrame,
        factor_name: str,
        **kwargs
        ) -> pd.DataFrame:
        """
        对全市场股票池计算指定因子，返回 long format DataFrame
        
        参数:
            data: 全市场股票日频数据 DataFrame，必须包含 date_col, stock_col 以及因子所需列
            factor_name: 因子名，例如 'alpha1'，对应类中的静态方法名
            date_col: 日期列名，默认 'date'
            stock_col: 股票代码列名，默认 'stock'
            **kwargs: 额外参数（如某些因子需要特殊参数）
        
        返回:
            DataFrame 列: date, stock_code, alpha_value
        """
        factor_func = getattr(cls, factor_name, None)
        if factor_func is None:
            raise ValueError(f"因子 {factor_name} 不存在于 Alpha_Set 中")
        alpha_values = factor_func(data, **kwargs)
        result = pd.DataFrame()
        result['date'] = data['date']
        result['stock_code'] = data['stock_code']
        result['alpha_value'] = alpha_values
        result.reset_index(drop=True)
        return result

    @classmethod
    def compute_all_factors(
        cls,
        data: pd.DataFrame,
        factor_list: List[str],
        parallel: bool = False,
        ) -> Dict[str, pd.DataFrame]:
        """
        批量计算多个因子，返回字典 {因子名: 因子值DataFrame}
        """
        results = {}
        for factor_name in factor_list:
            print(f"Computing {factor_name} ...")
            df = cls.compute_factor_for_universe(
                data, factor_name
            )
            results[factor_name] = df
        return results

    # 可选：提供一个将单股票因子合并为面板的函数
    @staticmethod
    def merge_factor_to_panel(factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        将 long format (date, stock, value) 转为 MultiIndex 面板:
        index = (date, stock), columns = 因子名（此处只有一列 value）
        实际使用时建议将多个因子分别计算后合并。
        """
        return factor_df.set_index(['date', 'stock'])['value']