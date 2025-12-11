"""
Action Space Definitions for EETA.

Defines the 5 trading actions available to the agent.
"""


class Actions:
    """Trading action definitions."""
    NO_TRADE = 0
    LONG_STOCK = 1
    SHORT_STOCK = 2
    LONG_VOL = 3      # Simulated straddle
    SHORT_VOL = 4     # Simulated iron condor
    
    NAMES = ['NO_TRADE', 'LONG', 'SHORT', 'LONG_VOL', 'SHORT_VOL']
    
    @classmethod
    def name(cls, action: int) -> str:
        """Get action name."""
        return cls.NAMES[action] if 0 <= action < len(cls.NAMES) else 'UNKNOWN'
    
    @classmethod
    def all_actions(cls) -> list:
        """Get list of all action values."""
        return [cls.NO_TRADE, cls.LONG_STOCK, cls.SHORT_STOCK, cls.LONG_VOL, cls.SHORT_VOL]
