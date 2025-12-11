"""
Action Simulator for EETA.

Simulates P&L for each action type, including volatility plays
using VIX-based expected move simulation.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np

from ..rl.actions import Actions

logger = logging.getLogger(__name__)


class ActionSimulator:
    """
    Simulates P&L for each action type.
    
    Actions:
    0: NO_TRADE - No position
    1: LONG_STOCK - Buy shares
    2: SHORT_STOCK - Short shares
    3: LONG_VOL - Long volatility (simulated straddle)
    4: SHORT_VOL - Short volatility (simulated iron condor)
    
    For volatility plays, we use VIX-based expected move simulation
    since historical options data is expensive.
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,         # 0.05% slippage
    ):
        """
        Initialize action simulator.
        
        Args:
            transaction_cost: Cost as fraction of trade value
            slippage: Slippage as fraction of trade value
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def simulate_pnl(
        self,
        action: int,
        actual_move: float,
        expected_move: float,
        position_size: float
    ) -> Dict[str, Any]:
        """
        Calculate P&L for a given action.
        
        Args:
            action: Action ID (0-4)
            actual_move: Actual price move (decimal, e.g., 0.05 for 5%)
            expected_move: VIX-implied expected move
            position_size: Position as fraction of portfolio
            
        Returns:
            Dictionary with P&L details
        """
        result = {
            'action': action,
            'action_name': Actions.name(action),
            'actual_move': actual_move,
            'expected_move': expected_move,
            'position_size': position_size,
            'gross_pnl_pct': 0.0,
            'costs': 0.0,
            'net_pnl_pct': 0.0,
            'rationale': '',
        }
        
        if action == Actions.NO_TRADE:
            result['rationale'] = "No position taken"
            return result
        
        # Calculate gross P&L based on action type
        if action == Actions.LONG_STOCK:
            gross_pnl = actual_move * position_size
            result['rationale'] = f"Long position: stock moved {actual_move:+.1%}"
            
        elif action == Actions.SHORT_STOCK:
            gross_pnl = -actual_move * position_size
            result['rationale'] = f"Short position: stock moved {actual_move:+.1%}"
            
        elif action == Actions.LONG_VOL:
            # Simulated straddle: profits when |actual| > expected
            # We pay "premium" equal to expected_move
            # We receive value equal to |actual_move|
            vol_pnl = abs(actual_move) - expected_move
            gross_pnl = vol_pnl * position_size
            result['rationale'] = (
                f"Long vol: |move|={abs(actual_move):.1%}, "
                f"expected={expected_move:.1%}, "
                f"net={vol_pnl:+.1%}"
            )
            
        elif action == Actions.SHORT_VOL:
            # Simulated iron condor: profits when |actual| < expected
            # We collect premium if move is small, lose if large
            threshold = expected_move * 0.7
            
            if abs(actual_move) < threshold:
                # Small move: collect premium
                vol_pnl = expected_move * 0.3  # Simplified premium
                result['rationale'] = (
                    f"Short vol WIN: |move|={abs(actual_move):.1%} < "
                    f"threshold={threshold:.1%}"
                )
            else:
                # Large move: lose
                # Loss is proportional to how much we exceeded threshold
                excess_move = abs(actual_move) - threshold
                vol_pnl = -excess_move * 0.5  # Simplified loss
                result['rationale'] = (
                    f"Short vol LOSS: |move|={abs(actual_move):.1%} > "
                    f"threshold={threshold:.1%}"
                )
            
            gross_pnl = vol_pnl * position_size
            
        else:
            logger.warning(f"Unknown action: {action}")
            result['rationale'] = f"Unknown action: {action}"
            return result
        
        # Calculate costs
        costs = (self.transaction_cost + self.slippage) * position_size
        
        # Net P&L
        net_pnl = gross_pnl - costs
        
        result['gross_pnl_pct'] = gross_pnl
        result['costs'] = costs
        result['net_pnl_pct'] = net_pnl
        
        return result
    
    def calculate_expected_move(self, vix: float, days: int = 1) -> float:
        """
        Calculate expected move based on VIX.
        
        This is our key formula for simulating options behavior.
        Expected Move = VIX/100 * sqrt(days/365)
        
        Args:
            vix: Current VIX level
            days: Days until event
            
        Returns:
            Expected move as decimal
        """
        return (vix / 100) * np.sqrt(days / 365)
    
    def calculate_straddle_breakeven(self, expected_move: float) -> Tuple[float, float]:
        """
        Calculate straddle breakeven points.
        
        Args:
            expected_move: Expected move (premium paid)
            
        Returns:
            Tuple of (lower_breakeven, upper_breakeven) as percentage moves
        """
        return (-expected_move, expected_move)
    
    def calculate_condor_max_profit(
        self,
        expected_move: float,
        premium_fraction: float = 0.3
    ) -> float:
        """
        Calculate iron condor maximum profit.
        
        Args:
            expected_move: Expected move
            premium_fraction: Fraction of expected move collected as premium
            
        Returns:
            Maximum profit as decimal
        """
        return expected_move * premium_fraction


class RewardCalculator:
    """
    Calculates reward for RL training.
    
    Design principles:
    1. Primary signal: Risk-adjusted P&L
    2. Penalize losses more than reward gains (risk aversion)
    3. Bonus for appropriate sizing
    4. Bonus for correct NO_TRADE decisions
    """
    
    def __init__(
        self,
        pnl_scale: float = 10.0,
        risk_aversion: float = 1.5,
        sizing_reward_weight: float = 0.1,
        correct_skip_bonus: float = 0.2,
        missed_opportunity_penalty: float = 0.05
    ):
        """
        Initialize reward calculator.
        
        Args:
            pnl_scale: Scale factor for P&L
            risk_aversion: Multiplier for losses (>1 means losses hurt more)
            sizing_reward_weight: Weight for sizing appropriateness bonus
            correct_skip_bonus: Bonus for correctly skipping bad trades
            missed_opportunity_penalty: Penalty for missing good trades
        """
        self.pnl_scale = pnl_scale
        self.risk_aversion = risk_aversion
        self.sizing_reward_weight = sizing_reward_weight
        self.correct_skip_bonus = correct_skip_bonus
        self.missed_opportunity_penalty = missed_opportunity_penalty
    
    # def calculate(
    #     self,
    #     action: int,
    #     pnl_pct: float,
    #     confidence: float,
    #     position_size: float,
    #     would_have_lost: bool = False
    # ) -> Tuple[float, Dict[str, float]]:
    #     """
    #     Calculate reward.
        
    #     Args:
    #         action: Action taken
    #         pnl_pct: Percentage P&L
    #         confidence: Pre-trade confidence
    #         position_size: Position size used
    #         would_have_lost: If NO_TRADE, would trading have lost?
            
    #     Returns:
    #         Tuple of (total_reward, components_dict)
    #     """
    #     components = {}
        
    #     # Component 1: Base P&L (scaled)
    #     base_reward = pnl_pct * self.pnl_scale
        
    #     # Apply risk aversion: losses hurt more
    #     if pnl_pct < 0:
    #         base_reward *= self.risk_aversion
        
    #     components['base_pnl'] = base_reward
        
    #     # Component 2: Sizing appropriateness
    #     # Reward for matching position size to confidence
    #     expected_size = self._confidence_to_size(confidence)
    #     size_diff = abs(position_size - expected_size)
    #     sizing_reward = (1 - size_diff * 5) * self.sizing_reward_weight
    #     components['sizing'] = sizing_reward
        
    #     # Component 3: Decision quality for NO_TRADE
    #     if action == Actions.NO_TRADE:
    #         if would_have_lost:
    #             # Correctly avoided a loss
    #             decision_reward = self.correct_skip_bonus
    #         else:
    #             # Missed an opportunity (small penalty)
    #             decision_reward = -self.missed_opportunity_penalty
    #     else:
    #         decision_reward = 0.0
        
    #     components['decision'] = decision_reward
        
    #     # Total reward
    #     total = base_reward + sizing_reward + decision_reward
        
    #     return total, components

    # def calculate(
    #     self,
    #     action: int,
    #     pnl_pct: float,
    #     confidence: float,
    #     position_size: float,
    #     would_have_lost: bool = False,
    #     actual_move: float = 0.0
    # ) -> Tuple[float, Dict[str, float]]:
    #     """
    #     Calculate reward with dynamic adjustments based on market opportunity.
        
    #     Args:
    #         action: Action taken
    #         pnl_pct: Percentage P&L
    #         confidence: Pre-trade confidence
    #         position_size: Position size used
    #         would_have_lost: If NO_TRADE, would trading have lost?
    #         actual_move: Actual stock move (for dynamic reward calculation)
            
    #     Returns:
    #         Tuple of (total_reward, components_dict)
    #     """
    #     components = {}
    #     move_magnitude = abs(actual_move)
        
    #     # Component 1: Base P&L (scaled)
    #     base_reward = pnl_pct * self.pnl_scale
        
    #     # Apply risk aversion: losses hurt more
    #     if pnl_pct < 0:
    #         base_reward *= self.risk_aversion
        
    #     components['base_pnl'] = base_reward
        
    #     # Component 2: Sizing appropriateness (only for actual trades)
    #     if action == Actions.NO_TRADE:
    #         sizing_reward = 0.0  # No sizing reward for not trading
    #     else:
    #         expected_size = self._confidence_to_size(confidence)
    #         size_diff = abs(position_size - expected_size)
    #         sizing_reward = (1 - size_diff * 5) * self.sizing_reward_weight
    #     components['sizing'] = sizing_reward

    #     # Component 3: DYNAMIC decision quality based on move magnitude
    #     if action == Actions.NO_TRADE:
    #         if move_magnitude < 0.01:
    #             # Small move (<1%) - good to stay out, no opportunity
    #             decision_reward = 0.02
    #         elif move_magnitude < 0.03:
    #             # Medium move (1-3%) - slight penalty for missing
    #             decision_reward = -0.03
    #         else:
    #             # Large move (>3%) - missed significant opportunity!
    #             decision_reward = -0.05 * (move_magnitude / 0.03)
    #     else:
    #         # Took action - small base bonus for engagement
    #         decision_reward = 0.01
            
    #         # Extra bonus for profitable trades on big moves
    #         if pnl_pct > 0 and move_magnitude > 0.03:
    #             decision_reward += 0.05
    #         # Extra penalty for losses on small moves (bad judgment)
    #         elif pnl_pct < 0 and move_magnitude < 0.02:
    #             decision_reward -= 0.02
        
    #     components['decision'] = decision_reward
        
    #     # Total reward
    #     total = base_reward + sizing_reward + decision_reward
        
    #     return total, components

    # def calculate(
    #     self,
    #     action: int,
    #     pnl_pct: float,
    #     confidence: float,
    #     position_size: float,
    #     would_have_lost: bool = False,
    #     actual_move: float = 0.0
    # ) -> Tuple[float, Dict[str, float]]:
    #     """
    #     SIMPLIFIED reward: PnL-dominant with clear positive/negative signals.
        
    #     Key principles:
    #     1. Profit = positive reward, Loss = negative reward (no offset!)
    #     2. NO_TRADE penalty = missed opportunity scaled by move size
    #     3. Correct direction should clearly beat wrong direction
    #     """
    #     components = {}
    #     move_magnitude = abs(actual_move)
        
    #     if action == Actions.NO_TRADE:
    #         # NO_TRADE: Penalty based on missed opportunity
    #         best_possible_pnl = move_magnitude * 0.02  # Assume 2% position size
            
    #         if move_magnitude < 0.015:
    #             # Small move (<1.5%) - OK to sit out
    #             reward = 0.0
    #         else:
    #             # Bigger move - penalize for missing opportunity
    #             reward = -best_possible_pnl * self.pnl_scale
            
    #         components['base_pnl'] = 0.0
    #         components['missed_opportunity'] = reward
    #         components['sizing'] = 0.0
    #         total = reward
            
    #     else:
    #         # TRADED: Reward = scaled P&L (positive or negative!)
    #         base_reward = pnl_pct * self.pnl_scale
            
    #         # Apply asymmetric risk aversion for losses
    #         if pnl_pct < 0:
    #             base_reward *= self.risk_aversion
            
    #         components['base_pnl'] = base_reward
    #         components['missed_opportunity'] = 0.0
            
    #         # Small sizing bonus
    #         expected_size = self._confidence_to_size(confidence)
    #         size_diff = abs(position_size - expected_size)
    #         sizing_reward = (1 - size_diff * 5) * self.sizing_reward_weight * 0.5
    #         components['sizing'] = sizing_reward
            
    #         total = base_reward + sizing_reward
        
    #     return total, components

    def calculate(
        self,
        action: int,
        pnl_pct: float,
        confidence: float,
        position_size: float,
        would_have_lost: bool = False,
        actual_move: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        SIMPLIFIED reward using actual_move directly for clear signal.
        
        Key principles:
        1. Correct direction = positive, Wrong direction = negative
        2. Reward scales with move magnitude
        3. NO_TRADE penalty for missing big moves
        """
        components = {}
        move_magnitude = abs(actual_move)
        reward_scale = 10.0  # Scale factor for rewards
        
        if action == Actions.NO_TRADE:
            # NO_TRADE: Penalty based on missed opportunity
            if move_magnitude < 0.015:
                # Small move (<1.5%) - OK to sit out
                reward = 0.0
            else:
                # Bigger move - penalize for missing opportunity
                reward = -move_magnitude * reward_scale * 0.5
            
            components['base_pnl'] = 0.0
            components['missed_opportunity'] = reward
            components['sizing'] = 0.0
            
        elif action == Actions.LONG_STOCK:
            # LONG: Reward = actual_move (positive if stock went up)
            reward = actual_move * reward_scale
            if reward < 0:
                reward *= self.risk_aversion  # Losses hurt more
            components['base_pnl'] = reward
            components['missed_opportunity'] = 0.0
            components['sizing'] = 0.0
            
        elif action == Actions.SHORT_STOCK:
            # SHORT: Reward = -actual_move (positive if stock went down)
            reward = -actual_move * reward_scale
            if reward < 0:
                reward *= self.risk_aversion  # Losses hurt more
            components['base_pnl'] = reward
            components['missed_opportunity'] = 0.0
            components['sizing'] = 0.0
            
        elif action == Actions.LONG_VOL:
            # LONG_VOL: Profits from big moves either direction
            # Reward = |actual_move| - expected_move (premium paid)
            expected_move = 0.04  # ~4% typical earnings move
            vol_pnl = move_magnitude - expected_move
            reward = vol_pnl * reward_scale
            if reward < 0:
                reward *= self.risk_aversion
            components['base_pnl'] = reward
            components['missed_opportunity'] = 0.0
            components['sizing'] = 0.0
            
        elif action == Actions.SHORT_VOL:
            # SHORT_VOL: Profits from small moves
            # Reward = expected_move - |actual_move| (capped)
            expected_move = 0.04
            vol_pnl = expected_move * 0.3 - max(0, move_magnitude - expected_move * 0.7) * 0.5
            reward = vol_pnl * reward_scale
            if reward < 0:
                reward *= self.risk_aversion
            components['base_pnl'] = reward
            components['missed_opportunity'] = 0.0
            components['sizing'] = 0.0
        
        else:
            reward = 0.0
            components['base_pnl'] = 0.0
            components['missed_opportunity'] = 0.0
            components['sizing'] = 0.0
        
        return reward, components

    def _confidence_to_size(self, confidence: float) -> float:
        """Map confidence to expected position size."""
        # Low confidence → small size, high confidence → larger size
        sizes = [0.005, 0.01, 0.02, 0.03, 0.05]
        idx = int(confidence * (len(sizes) - 1))
        idx = np.clip(idx, 0, len(sizes) - 1)
        return sizes[idx]
    
    def calculate_counterfactual(
        self,
        actual_move: float,
        expected_move: float,
        confidence: float
    ) -> Tuple[int, float]:
        """
        Calculate what the best action would have been.
        
        Useful for evaluating NO_TRADE decisions.
        
        Args:
            actual_move: What the stock actually did
            expected_move: Expected move
            confidence: Signal confidence
            
        Returns:
            Tuple of (optimal_action, hypothetical_pnl)
        """
        # Simulate each action
        simulator = ActionSimulator()
        size = self._confidence_to_size(confidence)
        
        best_action = Actions.NO_TRADE
        best_pnl = 0.0
        
        for action in [Actions.LONG_STOCK, Actions.SHORT_STOCK, 
                       Actions.LONG_VOL, Actions.SHORT_VOL]:
            result = simulator.simulate_pnl(
                action, actual_move, expected_move, size
            )
            if result['net_pnl_pct'] > best_pnl:
                best_pnl = result['net_pnl_pct']
                best_action = action
        
        return best_action, best_pnl
