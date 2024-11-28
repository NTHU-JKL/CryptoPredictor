from factor.factors import Factor
import pandas as pd
import numpy as np

class jiakai01(Factor):
    need = ["fear_greed_index"]

    def __init__(self, n: int = 5, threshold: float = 0.5):
        """
        初始化因子，n 是計算變化的時間窗口，threshold 是漲跌判斷的閾值。
        """
        self.n = n
        self.threshold = threshold

    def Gen(self, x: pd.DataFrame):
        """
        單筆資料預測漲跌（基於恐懼指數）。
        """
        for col in self.need:
            assert col in x.columns, f"{col} not exist"

        # 確保數據足夠
        fear_greed = x["fear_greed_index"]
        if len(fear_greed) < self.n + 1:
            return None  # 資料不足時無法預測

        # 計算變化率和波動率
        recent_return = (fear_greed.iloc[-1] - fear_greed.iloc[-self.n]) / fear_greed.iloc[-self.n]
        volatility = fear_greed.pct_change().rolling(self.n).std().iloc[-1]

        # 計算分數
        score = recent_return / (volatility + 1e-6)  # 避免分母為零
        if score > self.threshold:
            return 1  # 預測情緒樂觀（市場可能上漲）
        elif score < -self.threshold:
            return -1  # 預測情緒悲觀（市場可能下跌）
        else:
            return 0  # 預測情緒平穩（市場無明顯變化）

    def GenAll(self, x: pd.DataFrame) -> pd.Series:
        """
        整個資料庫批量處理，對每一個時間段進行預測。
        """
        for col in self.need:
            assert col in x.columns, f"{col} not exist"

        # 計算變化率和波動率
        fear_greed = x["fear_greed_index"]
        returns = (fear_greed - fear_greed.shift(self.n)) / fear_greed.shift(self.n)
        volatility = fear_greed.pct_change().rolling(self.n).std()

        # 計算分數並生成信號
        score = returns / (volatility + 1e-6)
        signals = pd.Series(0, index=x.index)  # 預設為 0（情緒平穩）
        signals[score > self.threshold] = 1  # 情緒樂觀
        signals[score < -self.threshold] = -1  # 情緒悲觀

        return signals

    def __str__(self) -> str:
        """
        返回因子的描述。
        """
        return f"{self.__class__.__name__}_n{self.n}_threshold{self.threshold}"

# 使用範例
if __name__ == "__main__":
    # 模擬恐懼指數數據
    data = {
        "fear_greed_index": [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]  # 假設的數據
    }
    df = pd.DataFrame(data)

    # 初始化因子
    jiakai_factor = jiakai01(n=3, threshold=0.5)