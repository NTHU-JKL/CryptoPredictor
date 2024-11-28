from factor.factors import Factor
import pandas as pd
import numpy as np

class jiakai02(Factor):
    need = ["fear_greed_index", "close"]

    def __init__(self, n_short: int = 5, n_long: int = 20, rsi_period: int = 14, threshold: float = 0.5):
        """
        初始化因子：
        - n_short: 短期移動平均線窗口
        - n_long: 長期移動平均線窗口
        - rsi_period: RSI 計算窗口
        - threshold: 信號生成的敏感度
        """
        self.n_short = n_short
        self.n_long = n_long
        self.rsi_period = rsi_period
        self.threshold = threshold

    def calculate_rsi(self, x: pd.Series) -> pd.Series:
        """
        計算 RSI（相對強弱指標）
        """
        delta = x.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()

        rs = avg_gain / (avg_loss + 1e-6)  # 避免除以零
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def Gen(self, x: pd.DataFrame):
        """
        單筆資料預測漲跌
        """
        for col in self.need:
            assert col in x.columns, f"{col} not exist"

        if len(x) < max(self.n_long, self.rsi_period) + 1:
            return None  # 資料不足無法預測

        # 計算移動平均線
        short_ma = x["close"].rolling(self.n_short).mean().iloc[-1]
        long_ma = x["close"].rolling(self.n_long).mean().iloc[-1]

        # 計算 RSI
        rsi = self.calculate_rsi(x["close"]).iloc[-1]

        # 獲取恐懼與貪婪指數
        fear_greed = x["fear_greed_index"].iloc[-1]

        # 判斷信號
        if fear_greed < 20 and short_ma > long_ma:
            return 1  # 買入信號（漲）
        elif fear_greed > 80 and rsi > 70:
            return -1  # 賣出信號（跌）

        return 0  # 無操作信號

    def GenAll(self, x: pd.DataFrame) -> pd.Series:
        """
        整個資料庫批量處理，返回每一個時間段的信號
        """
        for col in self.need:
            assert col in x.columns, f"{col} not exist"

        if len(x) < max(self.n_long, self.rsi_period) + 1:
            return pd.Series(0, index=x.index)  # 資料不足時返回全零

        # 計算移動平均線
        short_ma = x["close"].rolling(self.n_short).mean()
        long_ma = x["close"].rolling(self.n_long).mean()

        # 計算 RSI
        rsi = self.calculate_rsi(x["close"])

        # 獲取恐懼與貪婪指數
        fear_greed = x["fear_greed_index"]

        # 判斷信號
        buy_signals = (fear_greed < 20) & (short_ma > long_ma)
        sell_signals = (fear_greed > 80) & (rsi > 70)

        signals = pd.Series(0, index=x.index)  # 初始化為 0（無信號）
        signals[buy_signals] = 1  # 買入信號（漲）
        signals[sell_signals] = -1  # 賣出信號（跌）

        return signals

    def __str__(self) -> str:
        """
        返回因子的描述
        """
        return f"{self.__class__.__name__}_short{self.n_short}_long{self.n_long}_rsi{self.rsi_period}_threshold{self.threshold}"

# 使用範例
if __name__ == "__main__":
    # 模擬數據
    data = {
        "close": [100, 102, 104, 103, 105, 107, 110, 115, 120, 125, 130],
        "fear_greed_index": [10, 15, 18, 25, 30, 40, 60, 80, 85, 90, 95]
    }
    df = pd.DataFrame(data)

    # 初始化因子
    jiakai_factor = jiakai02(n_short=3, n_long=5, rsi_period=3, threshold=0.5)