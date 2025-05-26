"""
Inventory and quantity management for cost estimation.

This module provides classes for managing inventory and quantities:
- Inventory: Basic inventory management
- QuantityInputs: Advanced inventory profile management
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import param
try:
    import panel as pn
except ImportError:
    pn = None

# Get logger
logger = logging.getLogger(__name__)


class Inventory(param.Parameterized):
    """
    Class for managing program inventory and delivery schedules.
    
    Attributes:
        profile (pd.DataFrame): Inventory profile data
        delivery_cycle (int): Time between procurement and delivery
        service_life (int): Expected service life of items
    """
    
    profile = param.DataFrame(
        default=pd.DataFrame({
            'FY': range(2030, 2040),
            'quantity': [5]*5 + [10]*5
        })
    )
    delivery_cycle = param.Integer(2)
    service_life = param.Integer(20)

    def __init__(self, **params):
        super(Inventory, self).__init__(**params)
        df = self.profile.copy()
        if "delivery_cycle" not in df.columns:
            df = df.assign(delivery_cycle = self.delivery_cycle)
        else:
            df = df.assign(delivery_cycle = lambda x: x.delivery_cycle.fillna(self.delivery_cycle))
        if "service_life" not in df.columns:
            df = df.assign(service_life = self.service_life)
        else:
            df = df.assign(service_life = lambda x: x.service_life.fillna(self.service_life))
        # caclulate mid life
        df = df.assign(mid_life = lambda x: x.FY + x.delivery_cycle + x.service_life/2)
        # calculate delivery, retirement and inventory range
        self.profile = df


    @param.depends('profile', watch=True)
    def _update_profile(self) -> pd.DataFrame:
        """Update the profile."""
        logger.debug("Updating profile")
        df = self.profile.copy()
        df = df.assign(delivery = lambda x: x.FY + x.delivery_cycle)
        df = df.assign(retirement = lambda x: x.delivery+x.service_life)
        df = df.assign(inventory_range = lambda x: list(range(x.delivery, x.retirement)))

        self.profile = df
        return df

    @property
    def procurement(self) -> pd.DataFrame:
        """Get the procurement profile."""
        # add a column for delivery cycle and service life
        df = self.profile.copy()
        return df

    @property
    def delivery(self) -> pd.DataFrame:
        """Get the delivery schedule."""
        return self.procurement.assign(FY=lambda x: x.FY + x.delivery_cycle)

    @property
    def retirement(self) -> pd.DataFrame:
        """Get the retirement schedule."""
        return self.delivery.assign(
            FY=self.delivery.FY + x.service_life
            #retirements=self.quantity
        )

    
    @property
    def inventory(self) -> pd.DataFrame:
        """Get the inventory schedule."""
        df = self.delivery.copy()
        # calculate service life as  a range
        df = df.assign(FY = lambda x: list(range(x.FY, x.FY + self.service_life)))
        # explode the service life into separate rows
        df = df.explode('FY')
        # merge with retirement schedule
        return df
    
    def add_rows(self, n_rows: int, **kwargs) -> None:
        """Add rows to the inventory profile."""
        df = pd.DataFrame(index=range(n_rows), columns=self.profile.columns)
        #df.index.name = 'FY'
        for key, value in kwargs.items():
            df[key] = value
        self.profile = pd.concat([self.profile, df], ignore_index=True)
        #self._update_profile()

    def __panel__(self) -> Any:
        """Create a panel interface for the inventory data."""
        if pn is None:
            logger.warning("Panel is not installed, cannot create UI")
            return None
        pn.extension('tabulator')
        quantity_inputs = pn.Card(self.param.profile, title="Inputs")
        Inventory_outputs = pn.Card(lambda : self.inventory, title="Inventory", sizing_mode="stretch_width")
        return pn.Card(quantity_inputs, Inventory_outputs, title="Inventory", sizing_mode="stretch_width")


class QuantityInputs(param.Parameterized):
    """
    Class for managing quantity-based inputs and schedules.
    
    Attributes:
        procurement (pd.DataFrame): Procurement schedule
        delivery_cycle (int): Time between procurement and delivery
        service_life (int): Expected service life
        delivery (pd.DataFrame): Delivery schedule
        retirement (pd.DataFrame): Retirement schedule
        inventory (pd.DataFrame): Combined inventory profile
    """
    
    procurement = param.DataFrame(
        pd.DataFrame(columns=["FY", "Value"]),
        columns=set(["FY", "Value"])
    )
    delivery_cycle = param.Integer(2)
    service_life = param.Integer(20)
    delivery = param.DataFrame()
    retirement = param.DataFrame()
    inventory = param.DataFrame()
    
    def __init__(self, **params):
        """Initialize the quantity inputs."""
        super(QuantityInputs, self).__init__(**params)
        if not self.procurement.empty:
            self._calc_inventory()

    @param.depends('procurement', 'delivery_cycle', 'service_life', watch=True)
    def _calc_inventory(self) -> None:
        """
        Calculate inventory profiles from procurement data.
        
        This method creates delivery and retirement schedules
        based on the procurement profile and timing parameters.
        """
        logger.debug("Calculating inventory profiles")
        self.delivery = self.procurement.assign(
            FY=self.procurement.FY + self.delivery_cycle
        )
        self.retirement = self.delivery.assign(
            FY=self.delivery.FY + self.service_life
        )
        self.inventory = pd.concat([
            self.procurement.assign(Procurement=lambda x: x.Value).drop('Value', axis=1),
            self.delivery.assign(Delivery=lambda x: x.Value).drop(['Program', 'Value'], axis=1),
            self.retirement.assign(Retirement=lambda x: x.Value).drop(['Program', 'Value'], axis=1)
        ], axis=1)
        self.inventory[list(range(2020, 2050))] = np.nan 