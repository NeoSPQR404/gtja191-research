# Factor_Calculator.py
import pandas as pd
import numpy as np

class Factor_Calculator:
    """
    量化因子计算公共类（单资产时间序列版本）
    所有方法均假定输入为单只股票的时间序列（pandas Series 或 列），
    输出与输入等长的 Series，自动处理 NaN 和窗口不足的情况。
    若需横截面操作（如跨股票排名），请使用 CS_RANK 方法。
    """
    
    @staticmethod
    def RANK(A):
        """
        时间序列上的升序排名（沿自身时间轴）
        注意：原文档中的 RANK 本意为横截面排名，但多数 Alpha 公式中误用了时间序列排名。
        为保持与用户现有 Alpha_Set 一致，保留此实现。
        如需标准横截面排名，请使用 CS_RANK。
        """
        A = pd.Series(A).reset_index(drop=True)
        return A.rank()
    
    @staticmethod
    def CS_RANK(A):
        """横截面排名（适用于一维序列，如某日所有股票的因子值）"""
        A = pd.Series(A).reset_index(drop=True)
        return A.rank()
    
    @staticmethod
    def MAX(A, B):
        A = pd.Series(A).reset_index(drop=True)
        B = pd.Series(B).reset_index(drop=True)
        return np.maximum(A, B)
    
    @staticmethod
    def MIN(A, B):
        A = pd.Series(A).reset_index(drop=True)
        B = pd.Series(B).reset_index(drop=True)
        return np.minimum(A, B)
    
    @staticmethod
    def STD(A, n):
        A = pd.Series(A).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=max(1, n//2)).std()
        return res.fillna(0)
    
    @staticmethod
    def CORR(A, B, n):
        A = pd.Series(A).reset_index(drop=True)
        B = pd.Series(B).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=max(1, n//2)).corr(B)
        return res.fillna(0)
    
    @staticmethod
    def DELTA(A, n=1):
        A = pd.Series(A).reset_index(drop=True)
        return A - A.shift(n)
    
    @staticmethod
    def LOG(A):
        A = pd.Series(A).reset_index(drop=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(A)
    
    @staticmethod
    def SUM(A, n):
        A = pd.Series(A).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=1).sum()
        return res.fillna(0)
    
    @staticmethod
    def ABS(A):
        A = pd.Series(A).reset_index(drop=True)
        return np.abs(A)
    
    @staticmethod
    def MEAN(A, n):
        A = pd.Series(A).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=1).mean()
        return res.fillna(0)
    
    @staticmethod
    def TSRANK(A, n):
        """序列末位值在过去 n 天中的升序排名（1 ~ n）"""
        A = pd.Series(A).reset_index(drop=True)
        def _tsrank(x):
            if len(x) == 0:
                return np.nan
            return x.rank().iloc[-1]
        res = A.rolling(window=n, min_periods=n).apply(_tsrank, raw=False)
        return res.fillna(0)
    
    @staticmethod
    def SIGN(A):
        A = pd.Series(A).reset_index(drop=True)
        return np.sign(A)
    
    @staticmethod
    def COVIANCE(A, B, n):
        A = pd.Series(A).reset_index(drop=True)
        B = pd.Series(B).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=max(1, n//2)).cov(B)
        return res.fillna(0)
    
    @staticmethod
    def DELAY(A, n):
        A = pd.Series(A)
        res = A.shift(n)
        return res.fillna(0)
    
    @staticmethod
    def TSMIN(A, n):
        A = pd.Series(A).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=1).min()
        return res.fillna(0)
    
    @staticmethod
    def TSMAX(A, n):
        A = pd.Series(A).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=1).max()
        return res.fillna(0)
    
    @staticmethod
    def PROD(A, n):
        A = pd.Series(A).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=1).apply(np.prod, raw=True)
        return res.fillna(0)
    
    @staticmethod
    def COUNT(condition, n):
        condition = pd.Series(condition).astype(bool)
        res = condition.rolling(window=n, min_periods=1).sum()
        return res.fillna(0)
    
    @staticmethod
    def REGBETA(A, B, n):
        A = pd.Series(A).reset_index(drop=True)
        B = pd.Series(B).reset_index(drop=True)
        cov = A.rolling(window=n, min_periods=max(1, n//2)).cov(B)
        var = B.rolling(window=n, min_periods=max(1, n//2)).var()
        res = cov / var
        return res.fillna(0)
    
    @staticmethod
    def REGRESI(A, B, n):
        A = pd.Series(A).reset_index(drop=True)
        B = pd.Series(B).reset_index(drop=True)
        beta = Factor_Calculator.REGBETA(A, B, n)
        alpha = Factor_Calculator.MEAN(A, n) - beta * Factor_Calculator.MEAN(B, n)
        res = A - (alpha + beta * B)
        return res.fillna(0)
    
    @staticmethod
    def SMA(A, n, m):
        A = pd.Series(A).reset_index(drop=True)
        res = pd.Series(index=A.index, dtype=float)
        y_prev = np.nan
        for i in range(len(A)):
            if i == 0 or pd.isna(y_prev):
                res.iloc[i] = A.iloc[i]
            else:
                res.iloc[i] = (A.iloc[i] * m + y_prev * (n - m)) / n
            y_prev = res.iloc[i]
        return res.fillna(0)
    
    @staticmethod
    def SUMIF(A, n, condition):
        A = pd.Series(A).reset_index(drop=True)
        condition = pd.Series(condition).astype(bool)
        res = (A * condition).rolling(window=n, min_periods=1).sum()
        return res.fillna(0)
    
    @staticmethod
    def WMA(A, n):
        A = pd.Series(A).reset_index(drop=True)
        weights = np.array([0.9**i for i in range(n)][::-1])
        weights /= weights.sum()
        def wma(window):
            return np.average(window, weights=weights[-len(window):])
        res = A.rolling(window=n, min_periods=n).apply(wma, raw=True)
        return res.fillna(0)
    
    @staticmethod
    def DECAYLINEAR(A, d):
        A = pd.Series(A).reset_index(drop=True)
        weights = np.arange(d, 0, -1, dtype=float)
        weights /= weights.sum()
        result = A.rolling(window=d, min_periods=d).apply(
            lambda x: np.dot(x, weights[::-1]), raw=True)
        return result.fillna(0)
    
    @staticmethod
    def FILTER(A, condition):
        A = pd.Series(A).reset_index(drop=True)
        condition = pd.Series(condition).astype(bool)
        return A.where(condition, np.nan)
    
    @staticmethod
    def HIGHDAY(A, n):
        A = pd.Series(A).reset_index(drop=True)
        def _highday(x):
            return n - 1 - np.argmax(x)
        res = A.rolling(window=n, min_periods=n).apply(_highday, raw=True)
        return res.fillna(0)
    
    @staticmethod
    def LOWDAY(A, n):
        A = pd.Series(A).reset_index(drop=True)
        def _lowday(x):
            return n - 1 - np.argmin(x)
        res = A.rolling(window=n, min_periods=n).apply(_lowday, raw=True)
        return res.fillna(0)
    
    @staticmethod
    def SEQUENCE(n):
        return np.arange(1, n + 1)
    
    @staticmethod
    def SUMAC(A, n):
        A = pd.Series(A).reset_index(drop=True)
        res = A.rolling(window=n, min_periods=1).sum()
        return res.fillna(0)

# class Factor_Calculator:
#     """
#     量化因子计算公共类
    
#     所有方法均为静态方法，可直接通过类名调用，例如：
#         FactorCalculator.RANK(data['CLOSE'])
    
#     默认假设输入的数据集为 pandas DataFrame 列名通常包含
#     'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP', 'VOLUME', 'AMOUNT' 等。
#     """
    
#     @staticmethod
#     def RANK(A):
#         """向量A升序排序（返回升序排名）"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rank()
    
#     @staticmethod
#     def MAX(A, B):
#         """在A,B中选择最大的数"""
#         A = pd.Series(A).reset_index(drop=True)
#         B = pd.Series(B).reset_index(drop=True)
#         if isinstance(A, pd.Series) and isinstance(B, pd.Series):
#             return np.maximum(A, B)
#         return max(A, B)
    
#     @staticmethod
#     def MIN(A, B):
#         """在A,B中选择最小的数"""
#         A = pd.Series(A).reset_index(drop=True)
#         B = pd.Series(B).reset_index(drop=True)
#         if isinstance(A, pd.Series) and isinstance(B, pd.Series):
#             return np.minimum(A, B)
#         return min(A, B)
    
#     @staticmethod
#     def STD(A, n):
#         """序列A过去n天标准差"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).std()
    
#     @staticmethod
#     def CORR(A, B, n):
#         """序列A、B过去n天相关系数"""
#         A = pd.Series(A).reset_index(drop=True)
#         B = pd.Series(B).reset_index(drop=True)
#         return A.rolling(window=n).corr(B)
    
#     @staticmethod
#     def DELTA(A, n=1):
#         """A的一阶差分，DELTA(A,n)=A_t - A_{t-n}"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A - A.shift(n)
    
#     @staticmethod
#     def LOG(A):
#         """自然对数函数"""
#         A = pd.Series(A).reset_index(drop=True)
#         return np.log(A)
    
#     @staticmethod
#     def SUM(A, n):
#         """序列A过去n天求和"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).sum()
    
#     @staticmethod
#     def ABS(A):
#         """绝对值函数"""
#         A = pd.Series(A).reset_index(drop=True)
#         return np.abs(A)
    
#     @staticmethod
#     def MEAN(A, n):
#         """序列A过去n天均值"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).mean()
    
#     @staticmethod
#     def TSRANK(A, n):
#         """序列A的末位值在过去n天的顺序排位（升序排名）"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).apply(lambda x: x.rank().iloc[-1], raw=False)
    
#     @staticmethod
#     def SIGN(A):
#         """符号函数"""
#         A = pd.Series(A).reset_index(drop=True)
#         return np.sign(A)
    
#     @staticmethod
#     def COVIANCE(A, B, n):
#         """序列A、B过去n天协方差"""
#         A = pd.Series(A).reset_index(drop=True)
#         B = pd.Series(B).reset_index(drop=True)
#         return A.rolling(window=n).cov(B)
    
#     @staticmethod
#     def DELAY(A, n):
#         """滞后n期，即使用t-n时刻的值"""
#         A = pd.Series(A)
#         return A.shift(n)
    
#     @staticmethod
#     def TSMIN(A, n):
#         """序列A过去n天的最小值"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).min()
    
#     @staticmethod
#     def TSMAX(A, n):
#         """序列A过去n天的最大值"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).max()
    
#     @staticmethod
#     def PROD(A, n):
#         """序列A过去n天累乘"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).apply(np.prod, raw=True)
    
#     @staticmethod
#     def COUNT(condition, n):
#         """计算前n期满足条件condition的样本个数"""
#         condition = pd.Series(condition)
#         return condition.rolling(window=n).sum()
    
#     @staticmethod
#     def REGBETA(A, B, n):
#         """前n期样本A对B做回归所得回归系数（斜率）"""
#         A = pd.Series(A).reset_index(drop=True)
#         B = pd.Series(B).reset_index(drop=True)
#         cov = A.rolling(window=n).cov(B)
#         var = B.rolling(window=n).var()
#         return cov / var
    
#     @staticmethod
#     def REGRESI(A, B, n):
#         """前n期样本A对B做回归所得的残差"""
#         A = pd.Series(A).reset_index(drop=True)
#         B = pd.Series(B).reset_index(drop=True)
#         beta = Factor_Calculator.REGBETA(A, B, n)
#         alpha = Factor_Calculator.MEAN(A, n) - beta * Factor_Calculator.MEAN(B, n)
#         return A - (alpha + beta * B)
    
#     @staticmethod
#     def SMA(A, n, m):
#         """
#         计算SMA：Y_t = (A_t * m + Y_{t-1} * (n - m)) / n
#         其中Y表示最终结果
#         """
#         A = pd.Series(A).reset_index(drop=True)
#         res = pd.Series(index=A.index, dtype=float)
#         y_prev = np.nan
#         for i in range(len(A)):
#             if i == 0 or pd.isna(y_prev):
#                 res.iloc[i] = A.iloc[i]
#             else:
#                 res.iloc[i] = (A.iloc[i] * m + y_prev * (n - m)) / n
#             y_prev = res.iloc[i]
#         return res
    
#     @staticmethod
#     def SUMIF(A, n, condition):
#         """对A前n项条件求和，其中condition表示选择条件"""
#         A = pd.Series(A).reset_index(drop=True)
#         condition = pd.Series(condition)
#         return (A * condition).rolling(window=n).sum()
    
#     @staticmethod
#     def WMA(A, n):
#         """
#         计算A前n期样本加权平均值，权重为0.9^i（i表示样本距离当前时点的间隔，i=0为当前）
#         """
#         A = pd.Series(A).reset_index(drop=True)
#         weights = np.array([0.9**i for i in range(n)][::-1])
#         def wma(window):
#             return np.average(window, weights=weights[-len(window):])
#         return A.rolling(window=n).apply(wma, raw=True)
    
#     @staticmethod
#     def DECAYLINEAR(A, d):
#         """
#         对A序列计算移动平均加权，其中权重对应d, d-1, ..., 1（权重和为1）
#         """
#         A = pd.Series(A).reset_index(drop=True)
#         weights = np.arange(d, 0, -1, dtype=float)
#         weights /= weights.sum()
#         result = A.rolling(window=d, min_periods=d).apply(
#             lambda x: np.dot(x, weights[::-1]), raw=True)
#         return result
    
#     @staticmethod
#     def FILTER(A, condition):
#         """对A筛选出符合选择条件condition的样本"""
#         A = pd.Series(A).reset_index(drop=True)
#         condition = pd.Series(condition)
#         return A[condition]
    
#     @staticmethod
#     def HIGHDAY(A, n):
#         """计算A前n期时间序列中最大值距离当前时点的间隔（从0开始计数）"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).apply(lambda x: n - 1 - np.argmax(x), raw=True)
    
#     @staticmethod
#     def LOWDAY(A, n):
#         """计算A前n期时间序列中最小值距离当前时点的间隔"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).apply(lambda x: n - 1 - np.argmin(x), raw=True)
    
#     @staticmethod
#     def SEQUENCE(n):
#         """生成1到n的等差序列"""
#         return np.arange(1, n + 1)
    
#     @staticmethod
#     def SUMAC(A, n):
#         """计算A的前n项的累加（此处实现为过去n期滚动求和）"""
#         A = pd.Series(A).reset_index(drop=True)
#         return A.rolling(window=n).sum()