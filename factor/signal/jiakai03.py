from factor.factors import Factor
import pandas as pd
import numpy as np

class jiakai03(Factor):
    need = ["fear_greed_index"]

    def __init__(self, duration_high: int = 10, duration_low: int = 14, threshold_high: int = 80, threshold_low: int = 20):
        """
        初始化因子：
        - duration_high: 高恐懼指數（>80）持續時間條件
        - duration_low: 低恐懼指數（<20）持續時間條件
        - threshold_high: 高恐懼指數閾值
        - threshold_low: 低恐懼指數閾值
        """
        self.duration_high = duration_high
        self.duration_low = duration_low
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

    def Gen(self, x: pd.DataFrame):
        """
        單筆資料預測漲跌
        """
        for col in self.need:
            assert col in x.columns, f"{col} not exist"

        if len(x) < max(self.duration_high, self.duration_low):
            return None  # 資料不足無法預測

        # 獲取恐懼與貪婪指數
        fear_greed = x["fear_greed_index"]

        # 計算高持續性條件
        recent_high = (fear_greed.iloc[-self.duration_high:] > self.threshold_high).sum() >= self.duration_high
        # 計算低持續性條件
        recent_low = (fear_greed.iloc[-self.duration_low:] < self.threshold_low).sum() >= self.duration_low

        # 判斷信號
        if recent_low:
            return 1  # 底部反彈信號（漲）
        elif recent_high:
            return -1  # 市場頂部信號（跌）

        return 0  # 無操作信號

    def GenAll(self, x: pd.DataFrame) -> pd.Series:
        """
        整個資料庫批量處理，返回每一個時間段的信號
        """
        for col in self.need:
            assert col in x.columns, f"{col} not exist"

        if len(x) < max(self.duration_high, self.duration_low):
            return pd.Series(0, index=x.index)  # 資料不足時返回全零

        # 初始化信號
        signals = pd.Series(0, index=x.index)

        # 遍歷資料逐段計算信號
        for i in range(max(self.duration_high, self.duration_low), len(x)):
            recent_fear_greed = x["fear_greed_index"].iloc[:i]

            # 計算高持續性條件
            recent_high = (recent_fear_greed.iloc[-self.duration_high:] > self.threshold_high).sum() >= self.duration_high
            # 計算低持續性條件
            recent_low = (recent_fear_greed.iloc[-self.duration_low:] < self.threshold_low).sum() >= self.duration_low

            # 判斷信號
            if recent_low:
                signals.iloc[i] = 1  # 底部反彈信號（漲）
            elif recent_high:
                signals.iloc[i] = -1  # 市場頂部信號（跌）

        return signals

    def __str__(self) -> str:
        """
        返回因子的描述
        """
        return f"{self.__class__.__name__}_high{self.duration_high}_low{self.duration_low}_thresHigh{self.threshold_high}_thresLow{self.threshold_low}"

# 使用範例
if __name__ == "__main__":
    # 模擬數據
    data = {
        "fear_greed_index": [10, 15, 18, 25, 30, 85, 90, 95, 92, 89, 86, 80, 78, 15, 12, 10, 9, 8, 7, 5]
    }
    df = pd.DataFrame(data)

    # 初始化因子
    jiakai_factor = jiakai03(duration_high=10, duration_low=14)