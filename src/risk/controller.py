"""
Risk Controller for EETA.

Enforces hard risk limits that CANNOT be overridden by the RL agent.
This is a critical safety component for responsible trading.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, date
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeProposal:
    """Proposed trade from the agent."""
    ticker: str
    action: int
    action_name: str
    position_size: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskCheckResult:
    """Result of risk check."""
    approved: bool
    adjusted_trade: Optional[TradeProposal]
    reason: str
    warnings: List[str] = field(default_factory=list)


class RiskController:
    """
    Enforces hard risk limits that CANNOT be overridden by the RL agent.
    
    Risk Controls:
    - Maximum position size per trade
    - Daily loss limit
    - Maximum drawdown
    - Consecutive loss limit
    - Correlation limits (future)
    
    This controller can VETO any trade that violates limits.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.05,
        min_position_size: float = 0.005,
        daily_loss_limit: float = 0.03,
        max_drawdown: float = 0.10,
        consecutive_loss_limit: int = 5,
        initial_capital: float = 100000.0
    ):
        """
        Initialize risk controller.
        
        Args:
            max_position_size: Maximum position as fraction of portfolio (5%)
            min_position_size: Minimum position size (0.5%)
            daily_loss_limit: Maximum daily loss as fraction (3%)
            max_drawdown: Maximum drawdown from peak (10%)
            consecutive_loss_limit: Halt after this many consecutive losses
            initial_capital: Starting capital for tracking
        """
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.consecutive_loss_limit = consecutive_loss_limit
        
        # Portfolio state
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_start_capital = initial_capital
        self.current_date = None
        
        # Loss tracking
        self.consecutive_losses = 0
        
        # Trading state
        self.trading_halted = False
        self.halt_reason = None
        
        # History
        self.trade_history: List[Dict[str, Any]] = []
        self.risk_events: List[Dict[str, Any]] = []
    
    def check_trade(
        self,
        proposed_trade: TradeProposal
    ) -> RiskCheckResult:
        """
        Check if proposed trade passes all risk controls.
        
        Args:
            proposed_trade: Trade details from agent
            
        Returns:
            RiskCheckResult with approval status and adjustments
        """
        warnings = []
        adjusted_trade = TradeProposal(
            ticker=proposed_trade.ticker,
            action=proposed_trade.action,
            action_name=proposed_trade.action_name,
            position_size=proposed_trade.position_size,
            confidence=proposed_trade.confidence
        )
        
        # Check 1: Is trading halted?
        if self.trading_halted:
            self._log_risk_event("trade_rejected_halted", proposed_trade)
            return RiskCheckResult(
                approved=False,
                adjusted_trade=None,
                reason=f"Trading halted: {self.halt_reason}"
            )
        
        # Check 2: Daily loss limit
        if self._check_daily_loss_breach():
            self._halt_trading("Daily loss limit breached")
            self._log_risk_event("daily_loss_limit", proposed_trade)
            return RiskCheckResult(
                approved=False,
                adjusted_trade=None,
                reason="Daily loss limit reached"
            )
        
        # Check 3: Maximum drawdown
        if self._check_drawdown_breach():
            self._halt_trading("Maximum drawdown breached")
            self._log_risk_event("max_drawdown", proposed_trade)
            return RiskCheckResult(
                approved=False,
                adjusted_trade=None,
                reason="Maximum drawdown reached"
            )
        
        # Check 4: Consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self._halt_trading(f"Consecutive loss limit ({self.consecutive_loss_limit}) reached")
            self._log_risk_event("consecutive_losses", proposed_trade)
            return RiskCheckResult(
                approved=False,
                adjusted_trade=None,
                reason=f"Too many consecutive losses ({self.consecutive_losses})"
            )
        
        # Check 5: Position size limits
        if proposed_trade.position_size > self.max_position_size:
            warnings.append(
                f"Position size {proposed_trade.position_size:.1%} reduced to "
                f"max {self.max_position_size:.1%}"
            )
            adjusted_trade.position_size = self.max_position_size
        
        if proposed_trade.position_size < self.min_position_size:
            warnings.append(
                f"Position size {proposed_trade.position_size:.1%} increased to "
                f"min {self.min_position_size:.1%}"
            )
            adjusted_trade.position_size = self.min_position_size
        
        # Check 6: Reduce size if approaching limits
        size_reduction = self._calculate_size_reduction()
        if size_reduction < 1.0:
            original_size = adjusted_trade.position_size
            adjusted_trade.position_size *= size_reduction
            warnings.append(
                f"Position size reduced by {(1-size_reduction):.0%} due to risk limits"
            )
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Risk warning: {warning}")
        
        return RiskCheckResult(
            approved=True,
            adjusted_trade=adjusted_trade,
            reason="Passed all risk checks",
            warnings=warnings
        )
    
    def update_after_trade(
        self,
        pnl: float,
        trade_details: Dict[str, Any] = None
    ):
        """
        Update risk state after a trade completes.
        
        Args:
            pnl: P&L from the trade (as fraction)
            trade_details: Optional trade details for logging
        """
        # Update capital
        pnl_dollars = self.current_capital * pnl
        self.current_capital += pnl_dollars
        self.daily_pnl += pnl
        
        # Update peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Log trade
        self.trade_history.append({
            "timestamp": datetime.now(),
            "pnl": pnl,
            "pnl_dollars": pnl_dollars,
            "capital_after": self.current_capital,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "details": trade_details
        })
    
    def _check_daily_loss_breach(self) -> bool:
        """Check if daily loss limit is breached."""
        return self.daily_pnl < -self.daily_loss_limit
    
    def _check_drawdown_breach(self) -> bool:
        """Check if maximum drawdown is breached."""
        if self.peak_capital == 0:
            return False
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return drawdown > self.max_drawdown
    
    def _calculate_size_reduction(self) -> float:
        """
        Calculate position size reduction based on proximity to limits.
        
        Returns multiplier (0-1) to apply to position size.
        """
        # Check drawdown proximity
        if self.peak_capital > 0:
            current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
            dd_ratio = current_dd / self.max_drawdown
            
            # Start reducing at 50% of max drawdown
            if dd_ratio > 0.5:
                dd_reduction = 1 - (dd_ratio - 0.5)
            else:
                dd_reduction = 1.0
        else:
            dd_reduction = 1.0
        
        # Check daily loss proximity
        if self.daily_loss_limit > 0:
            daily_ratio = abs(min(0, self.daily_pnl)) / self.daily_loss_limit
            
            # Start reducing at 50% of daily limit
            if daily_ratio > 0.5:
                daily_reduction = 1 - (daily_ratio - 0.5)
            else:
                daily_reduction = 1.0
        else:
            daily_reduction = 1.0
        
        # Check consecutive losses
        if self.consecutive_loss_limit > 0:
            loss_ratio = self.consecutive_losses / self.consecutive_loss_limit
            
            # Start reducing at 50% of limit
            if loss_ratio > 0.5:
                loss_reduction = 1 - (loss_ratio - 0.5) * 0.5
            else:
                loss_reduction = 1.0
        else:
            loss_reduction = 1.0
        
        # Return minimum reduction
        return max(0.1, min(dd_reduction, daily_reduction, loss_reduction))
    
    def _halt_trading(self, reason: str):
        """Halt all trading activity."""
        self.trading_halted = True
        self.halt_reason = reason
        logger.warning(f"TRADING HALTED: {reason}")
        
        self._log_risk_event("trading_halted", None, {"reason": reason})
    
    def _log_risk_event(
        self,
        event_type: str,
        trade: Optional[TradeProposal],
        extra: Dict[str, Any] = None
    ):
        """Log a risk event for analysis."""
        self.risk_events.append({
            "timestamp": datetime.now(),
            "event_type": event_type,
            "trade": trade.__dict__ if trade else None,
            "capital": self.current_capital,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "extra": extra or {}
        })
    
    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.daily_pnl = 0.0
        self.daily_start_capital = self.current_capital
        self.current_date = date.today()
        
        # Only reset halt if it was due to daily limit
        if self.halt_reason == "Daily loss limit breached":
            self.trading_halted = False
            self.halt_reason = None
            logger.info("Daily reset: trading resumed")
    
    def reset_all(self):
        """Full reset of risk controller state."""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.daily_pnl = 0.0
        self.daily_start_capital = self.initial_capital
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = None
        self.trade_history = []
        self.risk_events = []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        drawdown = 0.0
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "drawdown": drawdown,
            "drawdown_limit": self.max_drawdown,
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_loss_limit": self.consecutive_loss_limit,
            "position_size_multiplier": self._calculate_size_reduction(),
            "total_trades": len(self.trade_history),
            "risk_events": len(self.risk_events)
        }
    
    def get_risk_report(self) -> str:
        """Generate a human-readable risk report."""
        status = self.get_status()
        
        report = [
            "=" * 50,
            "RISK CONTROLLER STATUS REPORT",
            "=" * 50,
            f"Trading Status: {'HALTED - ' + status['halt_reason'] if status['trading_halted'] else 'ACTIVE'}",
            "",
            "Capital:",
            f"  Current: ${status['current_capital']:,.2f}",
            f"  Peak: ${status['peak_capital']:,.2f}",
            f"  Drawdown: {status['drawdown']:.2%} (limit: {status['drawdown_limit']:.2%})",
            "",
            "Daily:",
            f"  P&L: {status['daily_pnl']:.2%} (limit: -{status['daily_loss_limit']:.2%})",
            "",
            "Risk Metrics:",
            f"  Consecutive Losses: {status['consecutive_losses']} (limit: {status['consecutive_loss_limit']})",
            f"  Position Size Multiplier: {status['position_size_multiplier']:.1%}",
            "",
            f"Total Trades: {status['total_trades']}",
            f"Risk Events: {status['risk_events']}",
            "=" * 50
        ]
        
        return "\n".join(report)
