def fibonacci(n, memo={}):
    if n < 0:
        raise ValueError('Input must be a non-negative integer')
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    elif n in memo:
        return memo[n]
    else:
        memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
        return memo[n]