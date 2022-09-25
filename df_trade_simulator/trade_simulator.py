from pandas import DataFrame, Series
from typing import Union

BUY, SELL, HODL = 'buy', 'sell', 'hodl'

class TradeSimulator():
    def __init__(self, df: DataFrame, **kwargs):
        def default(key, val):
            return val if key not in kwargs else kwargs[key]
        
        # main vars
        self.df = df.copy()
        self.fee = default('fee', 0.1) # kwargs.get('fee', 0.1)
        self.price_col = default('price_col', 'price')
        self.wallet_col = default('wallet_col', 'wallet')
        self.wallet_fee_col = default('wallet_fee_col', 'wallet_fee')
        self.roi_col = default('roi_col', 'roi')
        self.side_col = default('side_col', 'side')

    def auto_add_sides(self, step: int = 5000):
        df = self.df

        df[self.side_col] = float('nan')

        for n in range(0, len(df), step):
            df.loc[df.index[n], self.side_col] = BUY if n % (step * 2) == 0 else SELL

    def add_sides(self, buy: Series, sell: Series):
        self.df[self.side_col] = float('nan')
        self.df.loc[buy, self.side_col] = BUY
        self.df.loc[sell, self.side_col] = SELL

    def get_trades(self) -> DataFrame:
        # make shortcuts
        df = self.df
        # set last value of side column to hodl
        # just to observe the status of wallet at the last data point
        df.iloc[-1, df.columns.get_loc(self.side_col)] = HODL 
        trades = df[df[self.side_col].notnull()] # get rid of nans
        trades = trades.loc[trades[self.side_col] != trades[self.side_col].shift()].copy() # rm consecutive duplicates
        return trades

    def sim_wallet(self, fee: float = None):
        # set fee if needed
        self.fee = self.fee if fee is None else fee

        # make shortcuts
        df = self.df
        fee = self.fee

        # init vars
        _fee = (100 - fee) / 100
        init_wallet = 1    
        memory = {'price': None, 'wallet': [init_wallet], 'wallet_fee': [init_wallet], 'side': None}

        def roi(row, memory: dict) -> Union[Series, DataFrame]:
            # get last wallet values
            last_wallet = memory['wallet'][-1]
            last_wallet_fee = memory['wallet_fee'][-1]

            # check against same side to avoid duplicated side changes
            side = row[self.side_col]
            last_side = memory['side']

            # save in memory
            memory['side'] = side  

            # handle hodl
            if side == HODL:
                side = BUY if last_side == SELL else SELL

            if side == last_side:
                memory['wallet'].append(last_wallet)
                memory['wallet_fee'].append(last_wallet_fee)
                return float('nan')

            # get prices to calc roi
            price = row[self.price_col]
            last_price = memory['price']

            # save in memory
            memory['price'] = price

            # handle first value
            if last_price is None:
                return init_wallet
            
            # calc roi and wallet values
            roi = last_price / price if side == BUY else price / last_price
            wallet = last_wallet * roi
            wallet_fee = last_wallet_fee * roi * _fee

            # save in memory
            memory['wallet'].append(wallet)
            memory['wallet_fee'].append(wallet_fee)
            
            return roi

        # get trades a.k.a. side changes
        side_changed = self.get_trades()
        side_changed[self.roi_col] = side_changed.apply(lambda row: roi(row, memory), axis = 1)
        side_changed[self.wallet_col] = memory['wallet']
        side_changed[self.wallet_fee_col] = memory['wallet_fee']

        # pass calculated columns back
        for col in self.roi_col, self.wallet_col, self.wallet_fee_col:
            df[col] = side_changed[col]