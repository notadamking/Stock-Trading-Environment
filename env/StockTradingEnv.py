import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _nextObservation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.currentStep: self.currentStep +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.currentStep: self.currentStep +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.currentStep: self.currentStep +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.currentStep: self.currentStep +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.currentStep: self.currentStep +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.maxNetWorth / MAX_ACCOUNT_BALANCE,
            self.sharesHeld / MAX_NUM_SHARES,
            self.averageShareCost / MAX_SHARE_PRICE,
            self.totalSharesSold / MAX_NUM_SHARES,
            self.totalSalesValue / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _takeAction(self, action):
        currentPrice = random.uniform(
            self.df.loc[self.currentStep, "Open"], self.df.loc[self.currentStep, "Close"])

        actionType = action[0]
        amount = action[1]

        if actionType < 1:
            # buy amount * self.balance
            totalPossible = self.balance / currentPrice
            sharesBought = totalPossible * amount
            prevAvgShareCost = self.averageShareCost * self.sharesHeld
            avgAdditionalCost = sharesBought * currentPrice

            self.balance -= sharesBought * currentPrice
            self.averageShareCost = (
                prevAvgShareCost + avgAdditionalCost) / (self.sharesHeld + sharesBought)
            self.sharesHeld += sharesBought

        elif actionType < 2:
            # sell amount * self.sharesHeld
            sharesSold = self.sharesHeld * amount
            self.balance += sharesSold * currentPrice
            self.sharesHeld -= sharesSold
            self.totalSharesSold += sharesSold
            self.totalSalesValue += sharesSold * currentPrice

        netWorth = self.balance + self.sharesHeld * currentPrice

        if netWorth > self.maxNetWorth:
            self.maxNetWorth = netWorth

        if self.sharesHeld == 0:
            self.averageShareCost = 0

    def step(self, action):
        # Execute one time step within the environment
        self._takeAction(action)

        self.currentStep += 1

        if self.currentStep > len(self.df.loc[:, 'Open'].values) - 6:
            self.currentStep = 0

        delayModifier = (self.currentStep / MAX_STEPS)

        reward = self.balance * delayModifier
        done = self.balance <= 0 or self.balance > MAX_ACCOUNT_BALANCE

        obs = self._nextObservation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.maxNetWorth = INITIAL_ACCOUNT_BALANCE
        self.sharesHeld = 0
        self.averageShareCost = 0
        self.totalSharesSold = 0
        self.totalSalesValue = 0

        # Set the current step to a random point within the data frame
        self.currentStep = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._nextObservation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        currentPrice = self.df.loc[self.currentStep, "Open"]
        netWorth = self.balance + self.sharesHeld * currentPrice
        profit = netWorth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.currentStep}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.sharesHeld} (Total sold: {self.totalSharesSold})')
        print(
            f'Avg cost for held shares: {self.averageShareCost} (Total sales value: {self.totalSalesValue})')
        print(f'Net worth: {netWorth} (Max net worth: {self.maxNetWorth})')
        print(f'Profit: {profit}')
