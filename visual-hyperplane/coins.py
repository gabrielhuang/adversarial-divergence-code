def aux(amount, coins, n_coins):
    '''
    How many ways to combine n_coins taking values in coins to make amount?
    '''
    solutions = []
    if n_coins == 0:
        if amount == 0:
            return [[]]
        else:
            return []
    for coin in coins:
        sub_solutions = aux(amount-coin, coins, n_coins-1)
        for sol in sub_solutions:
            solutions.append([coin] + sol)
    return solutions
 
def aux_count(amount, coins, n_coins):
    '''
    How many ways to combine n_coins taking values in coins to make amount?
    '''
    solutions = 0
    if n_coins == 0:
        if amount == 0:
            return 1
        else:
            return 0
    for coin in coins:
        sub_solutions = aux_count(amount-coin, coins, n_coins-1)
        solutions += sub_solutions
    return solutions

# alias
generate_combinations = aux

if __name__ == '__main__':
    print '2 for 10', aux(10, range(10), 2)
    print '3 for 10', aux(10, range(10), 3)
    print '2 for 10', aux_count(10, range(10), 2)
    print '3 for 10', aux_count(10, range(10), 3)

    for i in xrange(2, 10):
        print '{} for {}'.format(i, 10), aux_count(10, range(10), i)
        print '{} for {}'.format(i, i*5), aux_count(i*5, range(10), i)
