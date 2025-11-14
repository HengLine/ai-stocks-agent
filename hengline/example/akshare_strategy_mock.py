#!/usr/bin/env python3
"""
AKShare策略示例 - 模拟数据版本
基于AKShare数据源的简单交易策略实现（使用模拟数据演示）
参考：https://akshare.akfamily.xyz/demo.html#id2
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
    用于演示策略逻辑
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
            
            # 生成随机价格走势
            np.random.seed(hash(stock_code) % 2**32)  # 确保每个股票的数据一致
            
            base_price = np.random.uniform(10, 50)
            returns = np.random.normal(0.001, 0.02, 250)
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1.0))  # 价格不能低于1
            
            # 生成OHLCV数据
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                high = close * np.random.uniform(1.0, 1.05)
                low = close * np.random.uniform(0.95, 1.0)
                open_price = low + (high - low) * np.random.random()
                volume = np.random.randint(1000000, 10000000)
                
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
        
        return {
            'symbol': stock_code,
            'name': f'股票{stock_code}',
            'current_price': latest['Close'],
            'change_percent': np.random.uniform(-5, 5),
            'volume': latest['Volume'],
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class AKShareStrategy:
    """
    AKShare策略类
    实现基于AKShare数据的简单交易策略
    """
    
    def __init__(self, initial_cash: float = 500000.0, use_mock: bool = True):
        """
        初始化策略
        
        Args:
            initial_cash: 初始资金
            use_mock: 是否使用模拟数据
        """
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.positions = {}  # 持仓信息 {stock_code: {'shares': int, 'avg_price': float}}
        self.trade_history = []  # 交易历史
        self.portfolio_value = []  # 组合价值历史
        
        # 初始化数据源
        if use_mock:
            self.akshare = MockAKShareSource()
            info("使用模拟AKShare数据源")
        else:
            from hengline.stock.sources.akshare_source import AKShareSource
            self.akshare = AKShareSource()
            info("使用真实AKShare数据源")
        
        # 策略参数
        self.max_position_pct = 0.2  # 单只股票最大持仓比例
        self.stop_loss_pct = 0.1    # 止损比例
        self.stop_profit_pct = 0.2  # 止盈比例
        self.ma_short = 5           # 短期均线
        self.ma_long = 20           # 长期均线
        
        info(f"AKShare策略初始化完成，初始资金: {initial_cash:,.2f}")
    
    def get_stock_data(self, stock_code: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            period: 时间周期
            
        Returns:
            股票数据DataFrame
        """
        try:
            data = self.akshare.get_stock_price_data(stock_code, period=period, interval="1d")
            if data is not None and not data.empty:
                # 计算技术指标
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
        """
        计算技术指标
        
        Args:
            data: 原始价格数据
            
        Returns:
            包含技术指标的数据
        """
        try:
            # 计算移动平均线
            data['MA5'] = data['Close'].rolling(window=self.ma_short).mean()
            data['MA20'] = data['Close'].rolling(window=self.ma_long).mean()
            
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
            
            return data
        except Exception as e:
            error(f"计算技术指标失败: {str(e)}")
            return data
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        生成交易信号
        
        Args:
            data: 股票数据
            
        Returns:
            交易信号字典
        """
        signals = {}
        
        if len(data) < self.ma_long:
            return signals
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # 均线交叉信号
        if (prev['MA5'] <= prev['MA20'] and 
            latest['MA5'] > latest['MA20'] and 
            latest['RSI'] < 70):
            signals['ma_cross'] = 'buy'
        elif (prev['MA5'] >= prev['MA20'] and 
              latest['MA5'] < latest['MA20'] and 
              latest['RSI'] > 30):
            signals['ma_cross'] = 'sell'
        
        # RSI超买超卖信号
        if latest['RSI'] < 30:
            signals['rsi_oversold'] = 'buy'
        elif latest['RSI'] > 70:
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
        
        return signals
    
    def calculate_position_size(self, stock_price: float) -> int:
        """
        计算建仓数量
        
        Args:
            stock_price: 股票价格
            
        Returns:
            建仓数量
        """
        max_position_value = self.current_cash * self.max_position_pct
        shares = int(max_position_value / stock_price / 100) * 100  # 整手
        return max(0, shares)
    
    def should_buy(self, stock_code: str, signals: Dict[str, str]) -> bool:
        """
        判断是否应该买入
        
        Args:
            stock_code: 股票代码
            signals: 交易信号
            
        Returns:
            是否买入
        """
        # 至少需要2个买入信号
        buy_signals = [k for k, v in signals.items() if v == 'buy']
        
        if len(buy_signals) >= 2:
            # 检查是否已持仓
            if stock_code not in self.positions:
                return True
            # 检查是否可以加仓
            elif self.positions[stock_code]['shares'] > 0:
                current_value = (self.positions[stock_code]['shares'] * 
                               self.get_current_price(stock_code))
                if current_value < self.current_cash * self.max_position_pct * 0.8:
                    return True
        
        return False
    
    def should_sell(self, stock_code: str, signals: Dict[str, str]) -> bool:
        """
        判断是否应该卖出
        
        Args:
            stock_code: 股票代码
            signals: 交易信号
            
        Returns:
            是否卖出
        """
        if stock_code not in self.positions:
            return False
        
        # 至少需要2个卖出信号
        sell_signals = [k for k, v in signals.items() if v == 'sell']
        
        if len(sell_signals) >= 2:
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
        """
        获取当前价格
        
        Args:
            stock_code: 股票代码
            
        Returns:
            当前价格
        """
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
        """
        执行买入操作
        
        Args:
            stock_code: 股票代码
            price: 买入价格
            shares: 买入数量
            
        Returns:
            是否成功
        """
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
        """
        执行卖出操作
        
        Args:
            stock_code: 股票代码
            price: 卖出价格
            shares: 卖出数量
            
        Returns:
            是否成功
        """
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
        """
        计算组合总价值
        
        Returns:
            组合总价值
        """
        total_value = self.current_cash
        
        for stock_code, position in self.positions.items():
            current_price = self.get_current_price(stock_code)
            if current_price > 0:
                total_value += current_price * position['shares']
        
        return total_value
    
    def run_strategy(self, stock_codes: List[str], days: int = 30) -> Dict:
        """
        运行策略
        
        Args:
            stock_codes: 股票代码列表
            days: 运行天数
            
        Returns:
            策略结果
        """
        info(f"开始运行AKShare策略，股票池: {stock_codes}, 运行天数: {days}")
        
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
        """
        获取策略结果
        
        Returns:
            策略结果字典
        """
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
    """
    主函数 - 运行AKShare策略示例
    """
    # 创建策略实例（使用模拟数据）
    strategy = AKShareStrategy(initial_cash=500000.0, use_mock=True)
    
    # 定义股票池（示例股票）
    stock_codes = ['600000', '000001', '000002', '600036', '000858']
    
    # 运行策略
    result = strategy.run_strategy(stock_codes, days=10)
    
    # 打印结果
    print("\n" + "="*50)
    print("AKShare策略回测结果（模拟数据）")
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
    
    print("\n最近交易记录:")
    for trade in result['trade_history'][-5:]:
        print(f"  {trade['datetime'].strftime('%Y-%m-%d %H:%M:%S')} "
              f"{trade['action']} {trade['stock_code']} "
              f"{trade['shares']}股 @ {trade['price']:.2f}")
    
    print("\n组合价值变化:")
    for record in result['portfolio_value_history'][-5:]:
        print(f"  {record['date'].strftime('%Y-%m-%d %H:%M:%S')} "
              f"价值: {record['value']:,.2f}, "
              f"现金: {record['cash']:,.2f}, "
              f"持仓: {record['positions']}")
    
    # 展示技术指标计算示例
    print("\n" + "="*50)
    print("技术指标示例（以600000为例）")
    print("="*50)
    data = strategy.get_stock_data('600000', period='3mo')
    if data is not None and not data.empty:
        latest = data.iloc[-1]
        print(f"最新价格: {latest['Close']:.2f}")
        print(f"MA5: {latest['MA5']:.2f}")
        print(f"MA20: {latest['MA20']:.2f}")
        print(f"RSI: {latest['RSI']:.2f}")
        print(f"MACD: {latest['MACD']:.4f}")
        print(f"MACD_Signal: {latest['MACD_Signal']:.4f}")
        print(f"布林带上轨: {latest['BB_Upper']:.2f}")
        print(f"布林带下轨: {latest['BB_Lower']:.2f}")
        
        # 生成交易信号示例
        signals = strategy.generate_signals(data)
        print(f"\n交易信号: {signals}")


if __name__ == "__main__":
    main()