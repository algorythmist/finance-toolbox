import unittest

from portfolio import *


class PortfolioTestCase(unittest.TestCase):

    def test_portfolio_return(self):
        r = [1, 2, 3]
        w = [3, 2, 1]
        total_return = portfolio_return(w, r)
        self.assertEqual(10, total_return)
        self.assertEqual(10, portfolio_return([1, 2, 3], [3, 2, 1]))


if __name__ == '__main__':
    unittest.main()
