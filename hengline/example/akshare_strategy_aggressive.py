#!/usr/bin/env python3
"""
AKShare策略示例 - 激进交易版本
降低交易门槛，展示更多交易信号和操作
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hengline.logger import info, warning, error, debug


class MockAKShareSource:
    """
    模拟AKShare数据源
    生成更活跃的价格数据用于演示
    """
    
    def __init__(self):
        self.stock_data = {}
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """生成模拟股票数据"""
        stock_codes = ['600000', '000001', '000002', '600036', '000858']
        
        for stock_code in stock_codes:
            # 生成250天的价格数据
            dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
            
            # 生成更活跃的价格走势
            np.random.seed(hash(stock_code) % 2**32)
            
            base_price = np.random.uniform(10, 50)
            prices = [base_price]
            
            # 生成更大的波动率
            for i in range(1, 250):
                trend = np.sin(i / 30) * 0.01  # 添加趋势性
                noise = np.random.normal(0, 0.03)  # 增加噪声
                ret = trend + noise
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1.0))
            
            # 生成OHLCV数据
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                high = close * np.random.uniform(1.0, 1.08)
                low = close * np.random.uniform(0.92, 1.0)
                open_price = low + (high - low) * np.random.random()
                volume = np.random.randint(1000000, 15000000)
                
                data.append({
                    'Date': date,
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(close, 2),
                    'Volume': volume
                })
            
            self.stock_data[stock_code] = pd.DataFrame(data).set_index('Date')
    
    def get_stock_price_data(self, stock_code: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """获取股票价格数据"""
        if stock_code not in self.stock_data:
            return None
        
        data = self.stock_data[stock_code].copy()
        
        # 根据period筛选数据
        if period == "1d":
            data = data.tail(1)
        elif period == "1wk":
            data = data.tail(5)
        elif period == "1mo":
            data = data.tail(20)
        elif period == "3mo":
            data = data.tail(60)
        elif period == "6mo":
            data = data.tail(120)
        elif period == "1y":
            data = data.tail(250)
        
        return data
    
    def get_stock_realtime_data(self, stock_code: str) -> Optional[Dict]:
        """获取实时数据"""
        if stock_code not in self.stock_data:
            return None
        
        data = self.stock_data[stock_code]
        latest = data.iloc[-1]
        
        # 添加一些随机变化
        current_price = latest['Close'] * np.random.uniform(0.98, 1.02)
        
        return {
            'symbol': stock_code,
            'name': f'股票{stock_code}',
            'current_price': round(current_price, 2),
            'change_percent': np.random.uniform(-8, 8),
            'volume': latest['Volume'],
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class AggressiveAKShareStrategy:
    """
    激进版AKShare策略类
    更容易触发交易信号，用于演示
    """
    
    def __init__(self, initial_cash: float = 500000.0, use_mock: bool = True):
        """
        初始化策略
        """
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.positions = {}  # 持仓信息
        self.trade_history = []  # 交易历史
        self.portfolio_value = []  # 组合价值历史
        
        # 初始化数据源
        if use_mock:
            self.akshare = MockAKShareSource()
            info("使用模拟AKShare数据源（激进模式）")
        else:
            from hengline.stock.sources.akshare_source import AKShareSource
            self.akshare = AKShareSource()
            info("使用真实AKShare数据源（激进模式）")
        
        # 更激进的策略参数
        self.max_position_pct = 0.3  # 单只股票最大持仓比例提高到30%
        self.stop_loss_pct = 0.15    # 止损比例放宽到15%
        self.stop_profit_pct = 0.25  # 止盈比例提高到25%
        self.ma_short = 3           # 缩短短期均线
        self.ma_long = 10           # 缩短短期均线
        self.min_buy_signals = 1    # 只需要1个买入信号
        self.min_sell_signals = 1   # 只需要1个卖出信号
        
        info(f"激进AKShare策略初始化完成，初始资金: {initial_cash:,.2f}")
    
    def get_stock_data(self, stock_code: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            data = self.akshare.get_stock_price_data(stock_code, period=period, interval="1d")
            if data is not None and not data.empty:
                data = self._calculate_technical_indicators(data)
                info(f"获取股票数据成功: {stock_code}, 数据量: {len(data)} 行")
                return data
            else:
                warning(f"获取股票数据失败: {stock_code}")
                return None
        except Exception as e:
            error(f"获取股票数据异常: {stock_code}, 错误: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # 计算移动平均线
            data['MA3'] = data['Close'].rolling(window=self.ma_short).mean()
            data['MA10'] = data['Close'].rolling(window=self.ma_long).mean()
            
            # 计算RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
            
            # 计算布林带
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # 添加更多指标
            data['Price_Change'] = data['Close'].pct_change()
            data['Volume_MA'] = data['Volume'].rolling(window=10).mean()
            
            return data
        except Exception as e:
            error(f"计算技术指标失败: {str(e)}")
            return data
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """生成交易信号"""
        signals = {}
        
        if len(data) < self.ma_long:
            return signals
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # 均线交叉信号
        if (prev['MA3'] <= prev['MA10'] and 
            latest['MA3'] > latest['MA10']):
            signals['ma_cross'] = 'buy'
        elif (prev['MA3'] >= prev['MA10'] and 
              latest['MA3'] < latest['MA10']):
            signals['ma_cross'] = 'sell'
        
        # RSI信号（更宽松的阈值）
        if latest['RSI'] < 35:  # 从30调整到35
            signals['rsi_oversold'] = 'buy'
        elif latest['RSI'] > 65:  # 从70调整到65
            signals['rsi_overbought'] = 'sell'
        
        # MACD信号
        if (prev['MACD'] <= prev['MACD_Signal'] and 
            latest['MACD'] > latest['MACD_Signal']):
            signals['macd_cross'] = 'buy'
        elif (prev['MACD'] >= prev['MACD_Signal'] and 
              latest['MACD'] < latest['MACD_Signal']):
            signals['macd_cross'] = 'sell'
        
        # 布林带信号
        if latest['Close'] <= latest['BB_Lower']:
            signals['bb_lower'] = 'buy'
        elif latest['Close'] >= latest['BB_Upper']:
            signals['bb_upper'] = 'sell'
        
        # 价格变化信号
        if latest['Price_Change'] < -0.03:  # 跌幅超过3%
            signals['price_drop'] = 'buy'
        elif latest['Price_Change'] > 0.03:  # 涨幅超过3%
            signals['price_rise'] = 'sell'
        
        # 成交量信号
        if latest['Volume'] > latest['Volume_MA'] * 1.5:
            if latest['Close'] > prev['Close']:
                signals['volume_up'] = 'buy'
            else:
                signals['volume_down'] = 'sell'
        
        return signals
    
    def calculate_position_size(self, stock_price: float) -> int:
        """计算建仓数量"""
        max_position_value = self.current_cash * self.max_position_pct
        shares = int(max_position_value / stock_price / 100) * 100  # 整手
        return max(0, shares)
    
    def should_buy(self, stock_code: str, signals: Dict[str, str]) -> bool:
        """判断是否应该买入"""
        buy_signals = [k for k, v in signals.items() if v == 'buy']
        
        if len(buy_signals) >= self.min_buy_signals:
            # 检查是否已持仓
            if stock_code not in self.positions:
                return True
            # 检查是否可以加仓（更宽松的条件）
            elif self.positions[stock_code]['shares'] > 0:
                current_value = (self.positions[stock_code]['shares'] * 
                               self.get_current_price(stock_code))
                if current_value < self.current_cash * self.max_position_pct * 0.9:
                    return True
        
        return False
    
    def should_sell(self, stock_code: str, signals: Dict[str, str]) -> bool:
        """判断是否应该卖出"""
        if stock_code not in self.positions:
            return False
        
        sell_signals = [k for k, v in signals.items() if v == 'sell']
        
        if len(sell_signals) >= self.min_sell_signals:
            return True
        
        # 检查止损止盈
        current_price = self.get_current_price(stock_code)
        avg_price = self.positions[stock_code]['avg_price']
        
        if current_price <= avg_price * (1 - self.stop_loss_pct):
            return True
        
        if current_price >= avg_price * (1 + self.stop_profit_pct):
            return True
        
        return False
    
    def get_current_price(self, stock_code: str) -> float:
        """获取当前价格"""
        try:
            realtime_data = self.akshare.get_stock_realtime_data(stock_code)
            if realtime_data and 'current_price' in realtime_data:
                return float(realtime_data['current_price'])
            
            # 如果实时数据获取失败，使用历史数据的最新价格
            data = self.get_stock_data(stock_code, period="1wk")
            if data is not None and not data.empty:
                return float(data['Close'].iloc[-1])
            
            return 0.0
        except Exception as e:
            error(f"获取当前价格失败: {stock_code}, 错误: {str(e)}")
            return 0.0
    
    def execute_buy(self, stock_code: str, price: float, shares: int) -> bool:
        """执行买入操作"""
        total_cost = price * shares
        
        if total_cost > self.current_cash:
            warning(f"资金不足，无法买入 {stock_code}")
            return False
        
        try:
            self.current_cash -= total_cost
            
            if stock_code in self.positions:
                # 加仓
                old_shares = self.positions[stock_code]['shares']
                old_avg_price = self.positions[stock_code]['avg_price']
                new_shares = old_shares + shares
                new_avg_price = ((old_shares * old_avg_price + shares * price) / new_shares)
                
                self.positions[stock_code] = {
                    'shares': new_shares,
                    'avg_price': new_avg_price
                }
            else:
                # 新建仓
                self.positions[stock_code] = {
                    'shares': shares,
                    'avg_price': price
                }
            
            # 记录交易
            self.trade_history.append({
                'datetime': datetime.now(),
                'action': 'buy',
                'stock_code': stock_code,
                'price': price,
                'shares': shares,
                'amount': total_cost
            })
            
            info(f"买入成功: {stock_code}, 价格: {price:.2f}, 数量: {shares}, 金额: {total_cost:.2f}")
            return True
            
        except Exception as e:
            error(f"买入失败: {stock_code}, 错误: {str(e)}")
            return False
    
    def execute_sell(self, stock_code: str, price: float, shares: int) -> bool:
        """执行卖出操作"""
        if stock_code not in self.positions:
            warning(f"未持仓 {stock_code}，无法卖出")
            return False
        
        if shares > self.positions[stock_code]['shares']:
            warning(f"持仓数量不足，无法卖出 {stock_code}")
            return False
        
        try:
            total_amount = price * shares
            self.current_cash += total_amount
            
            # 更新持仓
            remaining_shares = self.positions[stock_code]['shares'] - shares
            if remaining_shares <= 0:
                del self.positions[stock_code]
            else:
                self.positions[stock_code]['shares'] = remaining_shares
            
            # 记录交易
            self.trade_history.append({
                'datetime': datetime.now(),
                'action': 'sell',
                'stock_code': stock_code,
                'price': price,
                'shares': shares,
                'amount': total_amount
            })
            
            info(f"卖出成功: {stock_code}, 价格: {price:.2f}, 数量: {shares}, 金额: {total_amount:.2f}")
            return True
            
        except Exception as e:
            error(f"卖出失败: {stock_code}, 错误: {str(e)}")
            return False
    
    def calculate_portfolio_value(self) -> float:
        """计算组合总价值"""
        total_value = self.current_cash
        
        for stock_code, position in self.positions.items():
            current_price = self.get_current_price(stock_code)
            if current_price > 0:
                total_value += current_price * position['shares']
        
        return total_value
    
    def run_strategy(self, stock_codes: List[str], days: int = 15) -> Dict:
        """运行策略"""
        info(f"开始运行激进AKShare策略，股票池: {stock_codes}, 运行天数: {days}")
        
        for day in range(days):
            info(f"第 {day + 1} 天交易")
            
            for stock_code in stock_codes:
                try:
                    # 获取股票数据
                    data = self.get_stock_data(stock_code)
                    if data is None or data.empty:
                        continue
                    
                    # 生成交易信号
                    signals = self.generate_signals(data)
                    debug(f"{stock_code} 交易信号: {signals}")
                    
                    current_price = self.get_current_price(stock_code)
                    if current_price <= 0:
                        continue
                    
                    # 检查买入信号
                    if self.should_buy(stock_code, signals):
                        shares = self.calculate_position_size(current_price)
                        if shares > 0:
                            self.execute_buy(stock_code, current_price, shares)
                    
                    # 检查卖出信号
                    elif self.should_sell(stock_code, signals):
                        if stock_code in self.positions:
                            shares = self.positions[stock_code]['shares']
                            self.execute_sell(stock_code, current_price, shares)
                
                except Exception as e:
                    error(f"处理股票 {stock_code} 时出错: {str(e)}")
                    continue
            
            # 记录组合价值
            portfolio_value = self.calculate_portfolio_value()
            self.portfolio_value.append({
                'date': datetime.now(),
                'value': portfolio_value,
                'cash': self.current_cash,
                'positions': len(self.positions)
            })
            
            info(f"第 {day + 1} 天结束，组合价值: {portfolio_value:,.2f}, 现金: {self.current_cash:,.2f}, 持仓数: {len(self.positions)}")
        
        return self.get_strategy_result()
    
    def get_strategy_result(self) -> Dict:
        """获取策略结果"""
        final_value = self.calculate_portfolio_value()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        total_return_pct = total_return * 100
        
        # 计算交易统计
        buy_trades = [t for t in self.trade_history if t['action'] == 'buy']
        sell_trades = [t for t in self.trade_history if t['action'] == 'sell']
        
        result = {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(self.trade_history),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'current_positions': len(self.positions),
            'positions': self.positions.copy(),
            'trade_history': self.trade_history.copy(),
            'portfolio_value_history': self.portfolio_value.copy()
        }
        
        return result


def main():
    """主函数 - 运行激进AKShare策略示例"""
    # 创建激进策略实例
    strategy = AggressiveAKShareStrategy(initial_cash=500000.0, use_mock=True)
    
    # 定义股票池
    stock_codes = ['600000', '000001', '000002', '600036', '000858']
    
    # 运行策略
    result = strategy.run_strategy(stock_codes, days=15)
    
    # 打印结果
    print("\n" + "="*50)
    print("激进AKShare策略回测结果（模拟数据）")
    print("="*50)
    print(f"初始资金: {result['initial_cash']:,.2f}")
    print(f"最终价值: {result['final_value']:,.2f}")
    print(f"总收益率: {result['total_return_pct']:.2f}%")
    print(f"总交易次数: {result['total_trades']}")
    print(f"买入次数: {result['buy_trades']}")
    print(f"卖出次数: {result['sell_trades']}")
    print(f"当前持仓数: {result['current_positions']}")
    
    print("\n当前持仓:")
    for stock_code, position in result['positions'].items():
        current_price = strategy.get_current_price(stock_code)
        market_value = current_price * position['shares']
        profit_pct = ((current_price - position['avg_price']) / position['avg_price']) * 100
        print(f"  {stock_code}: {position['shares']}股, "
              f"成本价: {position['avg_price']:.2f}, "
              f"现价: {current_price:.2f}, "
              f"市值: {market_value:.2f}, "
              f"收益率: {profit_pct:.2f}%")
    
    print("\n所有交易记录:")
    for i, trade in enumerate(result['trade_history'], 1):
        print(f"  {i:2d}. {trade['datetime'].strftime('%Y-%m-%d %H:%M:%S')} "
              f"{trade['action']:4s} {trade['stock_code']} "
              f"{trade['shares']:4d}股 @ {trade['price']:7.2f} "
              f"金额: {trade['amount']:8.2f}")
    
    print("\n组合价值变化:")
    for i, record in enumerate(result['portfolio_value_history'], 1):
        print(f"  第{i:2d}天: {record['date'].strftime('%m-%d %H:%M')} "
              f"价值: {record['value']:10,.2f}, "
              f"现金: {record['cash']:10,.2f}, "
              f"持仓: {record['positions']:2d}")
    
    # 展示技术指标和信号示例
    print("\n" + "="*50)
    print("技术指标和信号示例（以000001为例）")
    print("="*50)
    data = strategy.get_stock_data('000001', period='1mo')
    if data is not None and not data.empty:
        latest = data.iloc[-1]
        print(f"最新价格: {latest['Close']:.2f}")
        print(f"MA3: {latest['MA3']:.2f}")
        print(f"MA10: {latest['MA10']:.2f}")
        print(f"RSI: {latest['RSI']:.2f}")
        print(f"MACD: {latest['MACD']:.4f}")
        print(f"MACD_Signal: {latest['MACD_Signal']:.4f}")
        print(f"布林带上轨: {latest['BB_Upper']:.2f}")
        print(f"布林带下轨: {latest['BB_Lower']:.2f}")
        print(f"价格变化: {latest['Price_Change']*100:.2f}%")
        print(f"成交量比率: {latest['Volume']/latest['Volume_MA']:.2f}")
        
        # 生成交易信号示例
        signals = strategy.generate_signals(data)
        print(f"\n交易信号: {signals}")
        
        buy_signals = [k for k, v in signals.items() if v == 'buy']
        sell_signals = [k for k, v in signals.items() if v == 'sell']
        print(f"买入信号: {buy_signals}")
        print(f"卖出信号: {sell_signals}")


if __name__ == "__main__":
    main()