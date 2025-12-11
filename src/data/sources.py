"""
Data Sources for EETA.

Handles fetching financial data from various free APIs:
- yfinance: Stock prices, earnings data, VIX
- Finnhub: News sentiment (optional)
"""

import os
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import finnhub
except ImportError:
    finnhub = None

logger = logging.getLogger(__name__)


class YFinanceSource:
    """
    Data source using yfinance for free financial data.
    
    Provides:
    - Stock OHLCV data
    - Earnings dates and EPS data
    - VIX historical data
    """
    
    def __init__(self, cache_dir: str = None):
        if yf is None:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        self.cache_dir = cache_dir
        self._ticker_cache = {}
    
    def get_ticker(self, symbol: str) -> 'yf.Ticker':
        """Get or create a cached ticker object."""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]
    
    def fetch_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical price data.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = self.get_ticker(ticker)
            history = stock.history(start=start_date, end=end_date, interval=interval)
            
            if history.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            history.columns = [c.lower().replace(' ', '_') for c in history.columns]
            history.index.name = 'date'
            
            return history
            
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame()
    
    # def fetch_earnings_history(self, ticker: str) -> pd.DataFrame:
        # """
        # Fetch historical earnings data.
        
        # Args:
        #     ticker: Stock symbol
            
        # Returns:
        #     DataFrame with earnings dates and EPS data
        # """
        # try:
        #     stock = self.get_ticker(ticker)
            
        #     # Get earnings history
        #     earnings = stock.earnings_history
            
        #     if earnings is None or len(earnings) == 0:
        #         logger.warning(f"No earnings history for {ticker}")
        #         return pd.DataFrame()
            
        #     # Standardize
        #     df = earnings.reset_index()
        #     df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
        #     # Rename for consistency
        #     rename_map = {
        #         'earnings_date': 'date',
        #         'quarter': 'date',
        #         'epsactual': 'eps_actual',
        #         'epsestimate': 'eps_estimate',
        #         'epsdifference': 'eps_difference',
        #         'surprisepercent': 'surprise_pct'
        #     }
        #     df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
        #     return df
            
        # except Exception as e:
        #     logger.error(f"Error fetching earnings for {ticker}: {e}")
        #     return pd.DataFrame()

    def fetch_earnings_history(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical earnings data with actual announcement dates.

        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with earnings dates and EPS data
        """
        try:
            stock = self.get_ticker(ticker)
            
            # Use earnings_dates which has actual announcement dates
            earnings = stock.earnings_dates
            
            if earnings is None or len(earnings) == 0:
                logger.warning(f"No earnings history for {ticker}")
                return pd.DataFrame()
            
            # Reset index to make earnings date a column
            df = earnings.reset_index()
            
            # Standardize column names
            df.columns = [c.lower().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
            
            # Rename for consistency
            rename_map = {
                'earnings_date': 'date',
                'eps_estimate': 'eps_estimate',
                'reported_eps': 'eps_actual',
                'surprise%': 'surprise_pct',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            # Remove timezone info for consistency
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            # Remove future earnings (no reported EPS yet)
            if 'eps_actual' in df.columns:
                df = df.dropna(subset=['eps_actual'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_vix(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch VIX (volatility index) data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with VIX data
        """
        return self.fetch_price_history("^VIX", start_date, end_date)
    
    def fetch_spy(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch SPY (S&P 500 ETF) data for benchmarking.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with SPY data
        """
        return self.fetch_price_history("SPY", start_date, end_date)
    
    def fetch_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company information.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with company info
        """
        try:
            stock = self.get_ticker(ticker)
            info = stock.info
            
            return {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'name': info.get('longName', ticker),
                'country': info.get('country', 'Unknown')
            }
            
        except Exception as e:
            logger.warning(f"Error fetching company info for {ticker}: {e}")
            return {'sector': 'Unknown', 'industry': 'Unknown', 'market_cap': 0}
    
    def calculate_earnings_move(
        self,
        prices: pd.DataFrame,
        earnings_date: datetime,
        lookback_days: int = 5,
        forward_days: int = 5
    ) -> Dict[str, float]:
        """
        Calculate price movement around an earnings date.
        
        Args:
            prices: DataFrame with price history
            earnings_date: Date of earnings announcement
            lookback_days: Days to look back
            forward_days: Days to look forward
            
        Returns:
            Dictionary with move statistics
        """
        if prices.empty:
            return {'move_pct': np.nan, 'pre_close': np.nan, 'post_close': np.nan}
        
        # Convert earnings_date to same type as index
        if isinstance(earnings_date, str):
            earnings_date = pd.to_datetime(earnings_date)
        
        # Ensure timezone compatibility
        if prices.index.tz is not None:
            earnings_date = earnings_date.tz_localize(prices.index.tz) if earnings_date.tz is None else earnings_date
        
        try:
            # Find closest trading days before and after
            pre_dates = prices.index[prices.index < earnings_date]
            post_dates = prices.index[prices.index >= earnings_date]
            
            if len(pre_dates) == 0 or len(post_dates) == 0:
                return {'move_pct': np.nan, 'pre_close': np.nan, 'post_close': np.nan}
            
            pre_date = pre_dates[-1]
            
            # Post date should be after earnings (next trading day if earnings after close)
            if len(post_dates) > 1:
                post_date = post_dates[1]  # Day after earnings
            else:
                post_date = post_dates[0]
            
            pre_close = prices.loc[pre_date, 'close']
            post_close = prices.loc[post_date, 'close']
            
            move_pct = (post_close - pre_close) / pre_close
            
            return {
                'move_pct': move_pct,
                'pre_close': pre_close,
                'post_close': post_close,
                'pre_date': pre_date,
                'post_date': post_date
            }
            
        except Exception as e:
            logger.debug(f"Error calculating earnings move: {e}")
            return {'move_pct': np.nan, 'pre_close': np.nan, 'post_close': np.nan}


    # def get_sp500_tickers(self) -> list:
    #     """Get list of S&P 500 tickers."""
    #     sp500_sample = [
    #         'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'JNJ', 'V',
    #         'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV', 'MRK', 'LLY', 'PEP', 'KO',
    #         'COST', 'AVGO', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT', 'DHR', 'NEE',
    #         'LIN', 'ADBE', 'NKE', 'TXN', 'PM', 'RTX', 'ORCL', 'CRM', 'AMD', 'INTC',
    #         'QCOM', 'IBM', 'GE', 'CAT', 'BA', 'HON', 'UPS', 'LOW', 'UNH', 'WFC'
    #     ]
    #     return sp500_sample

    def get_sp500_tickers(self) -> list:
        """Get list of S&P 500 tickers."""
        sp500_tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC',
            'CRM', 'ADBE', 'CSCO', 'ORCL', 'ACN', 'IBM', 'QCOM', 'TXN', 'AVGO', 'NOW',
            'INTU', 'AMAT', 'MU', 'LRCX', 'ADI', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
            'PNC', 'TFC', 'COF', 'BK', 'STT', 'CME', 'ICE', 'SPGI', 'MCO', 'AON',
            'MMC', 'AJG', 'TROW', 'NTRS', 'DFS', 'CFG', 'KEY', 'RF', 'HBAN', 'FRC',
            
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'HUM', 'MCK', 'CAH', 'ISRG', 'SYK',
            'BSX', 'MDT', 'ZTS', 'VRTX', 'REGN', 'MRNA', 'BIIB', 'IQV', 'IDXX', 'DXCM',
            
            # Consumer Discretionary
            'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG', 'MAR', 'HLT',
            'CMG', 'YUM', 'DG', 'DLTR', 'ORLY', 'AZO', 'ROST', 'DHI', 'LEN', 'PHM',
            'F', 'GM', 'APTV', 'EBAY', 'ETSY', 'BBY', 'KMX', 'GPC', 'POOL', 'TSCO',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'EL', 'KMB',
            'GIS', 'K', 'HSY', 'SJM', 'CAG', 'CPB', 'MKC', 'HRL', 'TSN', 'KHC',
            'MDLZ', 'STZ', 'BF.B', 'TAP', 'ADM', 'KR', 'SYY', 'WBA', 'CLX', 'CHD',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'PXD',
            'DVN', 'HES', 'HAL', 'BKR', 'FANG', 'MRO', 'APA', 'CTRA', 'OVV', 'TRGP',
            
            # Industrials
            'CAT', 'HON', 'UNP', 'UPS', 'BA', 'GE', 'RTX', 'LMT', 'DE', 'MMM',
            'FDX', 'EMR', 'ITW', 'ETN', 'PH', 'ROK', 'NSC', 'CSX', 'WM', 'RSG',
            'GD', 'NOC', 'TDG', 'CARR', 'OTIS', 'JCI', 'IR', 'SWK', 'FAST', 'PAYX',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM',
            'DOW', 'PPG', 'ALB', 'CF', 'MOS', 'IFF', 'FMC', 'CE', 'EMN', 'PKG',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'PSA', 'O', 'WELL', 'DLR', 'AVB',
            'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'PEAK', 'HST', 'KIM', 'REG',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES',
            'ED', 'AWK', 'DTE', 'ETR', 'FE', 'PPL', 'AEE', 'CMS', 'CNP', 'EVRG',
            
            # Communication Services
            'V', 'MA', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'ATVI',
            'EA', 'TTWO', 'WBD', 'PARA', 'FOX', 'FOXA', 'NWS', 'NWSA', 'LYV', 'MTCH'
        ]
        return sp500_tickers

class FinnhubSource:
    """
    Data source for Finnhub API (news and sentiment).
    
    Free tier: 60 API calls/minute
    """
    
    def __init__(self, api_key: str = None):
        if finnhub is None:
            raise ImportError("finnhub-python not installed. Run: pip install finnhub-python")
        
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.client = None
        
        if self.api_key:
            self.client = finnhub.Client(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if Finnhub API is available."""
        return self.client is not None
    
    def fetch_news(
        self,
        ticker: str,
        from_date: str,
        to_date: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch company news.
        
        Args:
            ticker: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of news articles
        """
        if not self.is_available():
            logger.warning("Finnhub API not available (no API key)")
            return []
        
        try:
            news = self.client.company_news(ticker, _from=from_date, to=to_date)
            return news or []
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def fetch_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch sentiment data for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Sentiment metrics
        """
        if not self.is_available():
            return {'buzz': 0, 'sentiment': 0}
        
        try:
            sentiment = self.client.news_sentiment(ticker)
            
            if sentiment:
                return {
                    'buzz': sentiment.get('buzz', {}).get('buzz', 0),
                    'sentiment': sentiment.get('sentiment', {}).get('score', 0),
                    'articles_in_week': sentiment.get('buzz', {}).get('articlesInLastWeek', 0)
                }
            
            return {'buzz': 0, 'sentiment': 0}
            
        except Exception as e:
            logger.debug(f"Error fetching sentiment for {ticker}: {e}")
            return {'buzz': 0, 'sentiment': 0}


class DataSourceManager:
    """
    Unified manager for all data sources.
    
    Provides a single interface for fetching data from multiple sources.
    """
    
    def __init__(
        self,
        cache_dir: str = None,
        finnhub_api_key: str = None
    ):
        self.yfinance = YFinanceSource(cache_dir)
        
        try:
            self.finnhub = FinnhubSource(finnhub_api_key)
        except ImportError:
            self.finnhub = None
            logger.info("Finnhub not available, news sentiment will be disabled")
    
    def fetch_all_for_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Fetch all available data for a ticker.
        
        Args:
            ticker: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with all data
        """
        result = {
            'ticker': ticker,
            'prices': self.yfinance.fetch_price_history(ticker, start_date, end_date),
            'earnings': self.yfinance.fetch_earnings_history(ticker),
            'info': self.yfinance.fetch_company_info(ticker)
        }
        
        if self.finnhub and self.finnhub.is_available():
            result['sentiment'] = self.finnhub.fetch_sentiment(ticker)
        else:
            result['sentiment'] = {'buzz': 0, 'sentiment': 0}
        
        return result
    
    def get_sp500_tickers(self) -> List[str]:
        """
        Get list of S&P 500 tickers.
        
        Returns:
            List of ticker symbols
        """
        # Common S&P 500 stocks for initial implementation
        # In production, could scrape from Wikipedia or use a data source
        sp500_sample = [
            # Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
            "ADBE", "NFLX", "PYPL", "CSCO", "ORCL", "IBM", "NOW", "QCOM", "TXN", "AVGO",
            # Finance
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
            "MA", "COF", "USB", "PNC", "TFC", "BK", "STT", "CME", "ICE", "SPGI",
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
            "AMGN", "GILD", "ISRG", "MDT", "CVS", "CI", "HUM", "SYK", "ZTS", "REGN",
            # Consumer
            "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
            "LOW", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "CL", "EL", "GIS",
            # Industrial
            "CAT", "HON", "UNP", "UPS", "BA", "GE", "MMM", "LMT", "RTX", "DE",
            "EMR", "ITW", "ETN", "PH", "ROK", "NSC", "CSX", "FDX", "WM", "RSG",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
            # Materials
            "LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "NUE", "VMC", "MLM",
            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB",
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ES", "WEC"
        ]
        
        return sp500_sample
