from math import sqrt, factorial
from decimal import *
import numpy as np
import numpy.random as npr
from copy import copy

def is_prime(n):
    if n <= 1:
        return False
    elif n % 2 == 0:
        if n == 2:
            return True
        return False
    for i in range(3, int(sqrt(n))+1, 2):
        if n % i == 0:
            return False
    return True

def prime_divisors(n):
    """return all prime divisors of n"""
    if n == 1:
        return []
    else:
        for p in range(2, n//2):
            if n % p == 0:
                return [p] + prime_divisors(n // p)
        return [n]

def fibonacci_list(n):
    f = []
    a = 1
    b = 1
    while a < n:
        f.append(a)
        a += b
        f.append(b)
        b += a
    if f[-1] > n:
        del(f[-1])
    return f

def n_first_primes(n):
    primes = [2]
    m = 3
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if m % p == 0:
                is_prime = False
        if is_prime:
            primes.append(m)
        m += 2
    return primes

def collatz_sequence(n):
    sequence = [n]
    x = n
    while x != 1:
        if x % 2 == 0:
            x //= 2
        else:
            x = 3 * x + 1
        sequence.append(x)
    return sequence

def lattice_paths(x, y):
    if x == 0 or y == 0:
        return 1
    if x == y:
        return 2 * lattice_paths(x -1, y)
    return lattice_paths(x - 1, y) + lattice_paths(x, y - 1)

def print_number_to_word(n):
    x =str(n)
    n2w = {
        1:'one',
        2:'two',
        3:'three',
        4:'four',
        5:'five',
        6:'six',
        7:'seven',
        8:'eight',
        9:'nine',
        10:'ten',
        11:'eleven',
        12:'twelve',
        13:'thirteen',
        14:'fourteen',
        15:'fifteen',
        16:'sixteen',
        17:'seventeen',
        18:'eighteen',
        19:'nineteen',
        20:'twenty',
        30:'thirty',
        40:'forty',
        50:'fifty',
        60:'sixty',
        70:'seventy',
        80:'eighty',
        90:'ninety',
        100:'hundred',
        1000:'thousand'
    }

    for i in range(len(x)):
        if i == len(x)-4:
            print(n2w[int(x[i])], 'thousand ', end='')
        elif x[len(x) - 3:len(x) - 1] == '100':
            print('one hundred')
            break
        elif i == len(x)-3 and x[i] != '0':
            print(n2w[int(x[i])], 'hundred and ', end='')
        elif i == len(x)-2 and x[i] != '0':
            if x[i] == '1':
                print(n2w[int(x[i:i + 2])])
                break
            else:
                print(n2w[int(x[i]+'0')]+'-', end='')
        elif i == len(x) -1 and x[i] != '0':
            print(n2w[int(x[i])])

def number_to_word(n):
    x =str(n)
    n2w = {
        1:'one',
        2:'two',
        3:'three',
        4:'four',
        5:'five',
        6:'six',
        7:'seven',
        8:'eight',
        9:'nine',
        10:'ten',
        11:'eleven',
        12:'twelve',
        13:'thirteen',
        14:'fourteen',
        15:'fifteen',
        16:'sixteen',
        17:'seventeen',
        18:'eighteen',
        19:'nineteen',
        20:'twenty',
        30:'thirty',
        40:'forty',
        50:'fifty',
        60:'sixty',
        70:'seventy',
        80:'eighty',
        90:'ninety',
        100:'hundred',
        1000:'thousand'
    }
    word = []
    for i in range(len(x)):
        if i == len(x)-4:
            word.append(n2w[int(x[i])] + ' thousand ')
        elif i == len(x)-3 and x[i] != '0':
            word.append(n2w[int(x[i])])
            if x[len(x) - 2:len(x)] == '00':
                word.append(' hundred')
                break
            else:
                word.append(' hundred and ')
        elif i == len(x)-2 and x[i] != '0':
            if x[i] == '1':
                word.append(n2w[int(x[i:i + 2])])
                break
            else:
                word.append(n2w[int(x[i]+'0')]+' ')
        elif i == len(x) -1 and x[i] != '0':
            word.append(n2w[int(x[i])])
    return ''.join(word)

def first_sundays(y):
    months =['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    month_len = {
        'jan':31,
        'feb':28,
        'mar':31,
        'apr':30,
        'may':31,
        'jun':30,
        'jul':31,
        'aug':31,
        'sep':30,
        'oct':31,
        'nov':30,
        'dec':31
    }
    total_days = 0
    total_first_sundays = 0
    for year in range(1900,y+1):
        for month in months:
            if year % 4 == 0 and year % 100 != 0:
                month_len['feb'] = 29
            else:
                month_len['feb'] = 28
            for day in range(month_len[month]):
                total_days += 1
                if day == 0 and total_days % 7 == 0:
                    total_first_sundays += 1
    return total_first_sundays

def unique_divisors(n):
    """return list of unique divisors of n including 1 and n"""
    divisors_list = []
    for i in range(1, (n//2)+1):
        if n % i == 0:
            divisors_list.append(i)
    return divisors_list + [n]

def longest_repeating():
    longest_pattern = 1
    d = 0
    getcontext().prec = 4000
    for n in range(2,1000):
        x = str(Decimal(1) / Decimal(n))[2:]
        pattern_length = 0
        for l in reversed(range(1,int(len(x)/2))):
            #for i in range(len(x)-2*l-1):
            #    if x[i:i+l] == x[i+l:i+2*l]:
            #        pattern_length = l
            if x[-l:] == x[-2*l:-l]:
                pattern_length = l
        if pattern_length > longest_pattern:
            print(Decimal(1)/Decimal(n), n, pattern_length)
            longest_pattern = pattern_length
            d = n
    return d

def create_numberspiral(n, spiral=None):
    if spiral is None:
        spiral = np.zeros((n,n), dtype=int)
    if n == 1:
        return 1
    else:
        x = n**2
        for i in reversed(range(1,n)):
            spiral[n-1, i] = x
            x -= 1
        for j in reversed(range(1,n)):
            spiral[j, 0] = x
            x -= 1
        for i in range(0,n-1):
            spiral[0, i] = x
            x -= 1
        for j in range(0,n-1):
            spiral[j, n-1] = x
            x -= 1
        #print(spiral)
        spiral[1:n-1, 1:n-1] = create_numberspiral(n-2, spiral[1:n-1, 1:n-1])
        return spiral

def is_pandigital(n:int):
    """Check if n contains all non-zero digits"""
    return set(digits(n)) == {1,2,3,4,5,6,7,8,9}

def factorial_digit_sum(n):
    """return sum of the factorial of each digit in n"""
    return sum([factorial(d) for d in digits(n)])

def number_rotations(n):
    rotations = []
    num = str(n)
    for i in range(len(num)):
        rotations.append(int(num[i:] + num[:i]))
    return rotations

def prime_sieve(r):
    P = np.arange(2, r+1, dtype=int)
    for i in range(r-1):
        p = P[i]
        if p != 0:
            P[i+p::p] = 0
    return P[P > 0]

def circular_primes():
    primes = prime_sieve(999999)
    n = 0
    for p in primes:
        circular = True
        for i in number_rotations(p):
            if i not in primes:
                circular = False
        if circular:
            print(p)
            n += 1
    print(n)

def double_palindrome():
    total = 0
    for i in range(1000000):
        d = str(i)
        b = str(bin(i))[2:]
        if d == d[::-1] and b == b[::-1]:
            print(d,b)
            total += i
    print(total)

def is_trunc_prime(n):

    def left_trunc(p):
        """Check if p is truncatable prime from the left."""
        if p < 10:
            return is_prime(p)
        else:
            if is_prime(p):
                decade = 10
                while p >= decade:
                    decade *= 10
                decade //= 10
                if left_trunc(p % decade):
                    return True
            else:
                return False

    def right_trunc(p):
        """Check if p is truncatable prime from the left."""
        if p < 10:
            return is_prime(p)
        else:
            if is_prime(p):
                if right_trunc(p // 10):
                    return True
            else:
                return False
    # If n is both left and right truncatable, then n is a truncatable prime
    return left_trunc(n) and right_trunc(n)

def find_all_truncatable_primes():
    all_trunc_primes = []
    i = 11
    while len(all_trunc_primes) < 11:
        if is_trunc_prime(i):
            print(i)
            all_trunc_primes.append(i)
        i += 2
    return all_trunc_primes

def pandigital_multiple():
    greatest_multiple = 1
    r = 10
    while r < 10000:
        concat_product = ''
        factors = []
        for n in range(1,r):
            product = str(r * n)
            is_concat = True
            for c in product:
                if c in concat_product:
                    is_concat = False
            if not is_concat:
                break
            else:
                factors.append(n)
                concat_product += product
            if is_pandigital(concat_product):
                print(r, factors, concat_product)
                greatest_multiple = r
                break
        r += 1
    print(greatest_multiple)

def circle_points(c):
    points = 0
    c_2 = c**2
    for a in range(1, c):
        b = sqrt(c_2 - a**2)
        if b == int(b):
            print(a, int(b), c)
            points += 1
    return points * 4

def int_triangles(p):
    used_numbers = set()
    triangles = 0
    a_max = (p-1)//2
    for a in range(1, a_max):
        if a not in used_numbers:
            c = (a**2/(p-a) + (p-a)) / 2
            if c == int(c):
                c = int(c)
                b = p - a - c
                used_numbers |= {a, b, c}
                triangles += 1
    return triangles

def int_triangle_max_solutions(p):
    max_solutions = 0
    max_sol_p = 0
    for i in range(p+1):
        s = int_triangles(i)
        print(max_sol_p)
        if s > max_solutions:
            max_solutions = s
            max_sol_p = i
    return max_sol_p, max_solutions

def nth_digit(n):
    """Finding the nth digit in the concatenation of all positive integers."""

    # Number of digits already passed
    d = 0

    # Iterating integer
    i = 0

    while d < n:
        i += 1
        d += len(str(i))
    return int(str(i)[n-d-1])

def alpha_index(c: str):
    alpha = "abcdefghijklmnopqrstuvwxyz"
    return alpha.index(c.lower()) + 1

def word_sum(w):
    s = 0
    for c in w:
        s += alpha_index(c)
    return s

def prime_append(primes: list):
    n = primes[-1]
    found = False
    while not found:
        n += 2
        found = True
        for p in primes:
            if n % p == 0:
                found = False
                break
    return primes + [n]

def is_square(n):
    x = sqrt(n)
    return x == int(x)

def prime_family(num: str):
    family = []
    for d in range(10):
        n = int(num.replace('*',str(d)))
        if is_prime(n):
            family.append(n)
    return family

def num_star_perm(n):
    if n == 1:
        return ['1','3','7','9']
    else:
        perm = []
        for p in num_star_perm(n-1):
            for d in [str(d) for d in range(10)] + ['*']:
                perm.append(d + p)
        return perm #[p for p in perm if '*' in p]

def digits(n):
    """return a list of the digits in the number n"""
    d = []
    r = n
    while r > 10:
        d.append(r % 10)
        r //= 10
    return (d + [r])[::-1]

def sup_decade(n):
    decade = 1
    while n // decade >= 1:
        decade *= 10
    return decade

def binomial(n, r):
    num = 1
    denum = 1
    for p in range(r+1,n+1):
        num *= p
        denum *= (n - p + 1)
    return num//denum

def permute_string(s: str):
    """Take a number and return all permutation of the digits"""
    if len(s) <= 1:
        return [s]
    else:
        permutations = []
        for i in range(len(s)):
            char = s[i]
            rest = s[0:i] + s[i+1:]
            for rest_perm in permute_string(rest):
                permutations.append(char + rest_perm)
        return list(set(permutations))

def permute_num(n: int):
    """Take a number and return all permutation of the digits"""
    return sorted([int(p)for p in permute_string(str(n))])

def permute_prime(prime: int):
    """Take a prime number and return all prime permutation of the digits"""
    return sorted([int(p) for p in permute_string(str(prime)) if is_prime(int(p))])

def permutation_check(permutations):
    """Check the permutations for three numbers with same difference"""
    if len(permutations) >= 4 and permutations[0] > 1000:
        for i, prime1 in enumerate(permutations[:-2]):
            for prime2 in permutations[i+1:-1]:
                prime3 = 2 * prime2 - prime1
                if prime3 in permutations:
                    return [prime1, prime2, prime3]
    return []

def find_prime_permutation():
    all_primes = primes_under(9999)
    for p in all_primes:
        if p > 1000:
            permutations = permute_prime(p)
            if permutation_check(permutations):
                print(permutation_check(permutations))

def is_palindromic(x: str):
    if len(x) <= 1:
        return True
    return x[0] == x[-1] and is_palindromic(x[1:-1])

def reversed_num(n):
    return int(str(n)[::-1])

def is_lychrel(n):
    x = n + reversed_num(n)
    for i in range(50):
        if is_palindromic(str(x)):
            return False
        x += reversed_num(x)
    return True

def sqrt_2(n, first=True):
    if n == 0:
        return int(first)
    else:
        return 1/(2 + sqrt_2(n-1, False)) + int(first)

def reduce_fraction(d, n):
    """:return reduced fraction of q/p"""
    if d == 1 or n == 1:
        return d, n
    elif is_prime(d) or is_prime(n):
        if d > n:
            if d % n == 0:
                return d // n, 1
            else:
                return d, n
        else:
            if n % d == 0:
                return 1, n // d
            else:
                return d, n
    q = d
    p = n
    i = 2
    while i <= q // 2 or i <= p // 2:
        if q % i == 0 and p % i == 0:
            q //= i
            p //= i
        else:
            i += 1
    if q == p:
        return 1, 1
    else:
        return q, p

def combine_2_fractions(a,b):
    """Combine 2 fractions"""
    q = a[0] * b[1] + a[1] * b[0]
    p = a[1] * b[1]
    return reduce_fraction(q, p)

def combine_simple(a,b):
    """Combine 2 fractions"""
    q = a[0] * b[1] + a[1] * b[0]
    p = a[1] * b[1]
    return q, p

def fraction_sum(*args):
    """Combine multiple fraction"""
    if len(args) == 2:
        return combine_2_fractions(args[0],args[-1])
    else:
        return combine_2_fractions(args[0], fraction_sum(*args[1:]))

def sqrt_2_fraction(n, first=True):
    if n == 0:
        return int(first), 1
    else:
        x = combine_simple((2,1),(sqrt_2_fraction(n-1, False)))
        return x[1] + int(first) * x[0], x[0]

def pair_check(s):
    for p1 in s:
        for p2 in s - {p1}:
            if not is_prime(int(str(p1) + str(p2))):
                return False
    return True

def prime_pair_sets():
    # {13, 5197, 5701, 8389, 6733} 26033
    primes = [3]
    sets = [{3}]
    found = False
    i = 5
    while not found:
        is_prime = True
        for p in primes:
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            for s in sets:
                if i not in s and pair_check(s | {i}):
                    print(s | {i})
                    sets.append(s | {i})
                    if len(s) == 4:
                        print(s | {i}, sum(s | {i}))
                        return s | {i}
            sets.append({i})
        i += 2

def is_pent(n):
    x = (1/2 + sqrt(1/4 + 6 * n))/3
    return x == int(x)

def is_cube(n):
    x = 1
    while x ** 3 < n:
        x += 1
    return x ** 3 == n

def cubic_permutations2():
    cubes = [i**3 for i in range(1, 10000)]
    for cube in cubes:
        other_cubes = [c for c in cubes if c > cube // 10 and c != cube]
        cube_digits = sorted(digits(cube))
        cube_permutations = [cube]

        for other_cube in other_cubes:
            if sorted(digits(other_cube)) == cube_digits:
                cube_permutations.append(other_cube)

        if len(cube_permutations) == 5:
            print(cube_permutations, min(cube_permutations))
            break

        print(cube, cube_permutations)

def solve_sqrt_fraction(x):
    """Find the terms of the continued fraction of an irrational square root"""

    # Finding the first term a0
    a0 = 0
    while (a0 + 1)**2 <= x:
        a0 += 1

    if a0**2 == x:
        return a0, None

    # Finding the repeated block of terms in the fraction
    # The general for of the fraction at index n is:
    #
    #          sqrt(x) - b(n)   |  a(n+1) is the integer part of the fraction
    #   a(n) + --------------   |                    x - b(n)**2
    #             c(n)          |  b(n+1) = a(n+1) * -----------  - b(n)
    #                           |                        c(n)
    #          sqrt(x) - b(n+1) |            x - bn**2
    # a(n+1) + --------------   |  c(n+1) = ----------
    #             c(n+1)        |              c(n)

    repeated_block = []
    bn = a0
    cn = 1
    block_complete = False
    while not block_complete:
        an = (cn * (a0 + bn)) // (x - bn ** 2)

        bn_plus1 = (an * (x - bn**2)) // cn - bn

        cn = (x - bn**2) // cn
        bn = bn_plus1

        # If the triple (an, bn, cn) already exists then the block is repeated
        if (an,bn,cn) in repeated_block:
            block_complete = True
        else:
            repeated_block.append((an, bn, cn))

    return a0, [i[0] for i in repeated_block]

def e_fraction(n, first=True, mod=1):
    if first:
        if n == 0:
            return 2, 1
        else:
            rest = e_fraction(n - 1, first=False, mod=mod+1)
            return 2 * rest[0] + rest[1], rest[0]
    else:
        if mod % 3 == 0:
            x = 2 * mod // 3
        else:
            x = 1

        if n == 0:
            return x, 1
        else:
            rest = e_fraction(n - 1, first=False, mod=mod + 1)
            return x * rest[0] + rest[1], rest[0]

def string_to_int_list(s):
    l = []
    n = ''
    for c in s:
        if c == ' ' or c == ',':
            l.append(int(n))
            n = ''
        else:
            n += c
    if n:
        l.append(int(n))
    return l

def import_triangle(filename):
    f = open(filename, 'r')
    triangle = []
    for line in f:
        triangle.append(string_to_int_list(line))
    f.close()
    return triangle

def import_matrix(filename):
    f = open(filename, 'r')
    matrix = []
    for line in f:
        matrix.append(string_to_int_list(line))
    f.close()
    return matrix

def max_path_sum_triangle(t):
    """ Calculates the longest path from top to bottom of triangle t using Dijkstra's algorithm.
        Since we want to find the longest path the weights have been normalised by subtracting each weight from 100
        And thus we can use Dijkstra's algorithm for finding the shortest path"""

    # List of all paths. Each element is a tuple with the path and its total length.
    paths = [([(0, 0)], 100 - t[0][0])]

    # Loops while there are uncompleted paths.
    while paths:

        # Unpacking the longest path
        long_path, long_length = paths.pop()

        last_node = long_path[-1]


        # Success condition
        if last_node[0] == len(t) - 1:
            print(100*len(t) - long_length, long_path)
            return long_path, 100 * len(t) - long_length


        # Creating two new paths by expanding to the next two nodes
        left_node = tuple(map(lambda x, y: x + y, last_node, (1, 0)))
        right_node = tuple(map(lambda x, y: x + y, last_node, (1, 1)))

        left_length  = long_length + 100 - t[left_node[0]][left_node[1]]

        right_length = long_length + 100 - t[right_node[0]][right_node[1]]

        # Comparing length of new paths for inserting them in the list of paths
        if left_length < right_length:
            new_short_path = (long_path + [left_node], left_length)
            new_long_path  = (long_path + [right_node], right_length)
        else:
            new_short_path = (long_path + [right_node], right_length)
            new_long_path  = (long_path + [left_node], left_length)


        SHORT_PATH_INSERTED = False
        LONG_PATH_INSERTED  = False
        i = len(paths) - 1
        while i >= 0:
            # if two paths visit the same node only the shortest path is kept
            if new_short_path[0][-1] == paths[i][0][-1]:
                if new_short_path[1] < paths[i][1]:
                    paths.pop(i)
                else:
                    SHORT_PATH_INSERTED = True

            if new_long_path[0][-1] == paths[i][0][-1]:
                if new_long_path[1] < paths[i][1]:
                    paths.pop(i)
                else:
                    LONG_PATH_INSERTED = True

            # the paths are inserted in diminishing order of length
            if not SHORT_PATH_INSERTED and paths[i][1] > new_short_path[1]:
                paths.insert(i+1, new_short_path)
                SHORT_PATH_INSERTED = True

            if not LONG_PATH_INSERTED and paths[i][1] > new_long_path[1]:
                paths.insert(i+1, new_long_path)
                LONG_PATH_INSERTED = True

            i -= 1

        # if no paths are longer, the new paths are inserted at the beginning
        if not SHORT_PATH_INSERTED:
            paths.insert(0, new_short_path)
        if not LONG_PATH_INSERTED:
            paths.insert(0, new_long_path)

def min_path_sum_matrix(m):
    # List of all paths. Each element is a tuple with the path and its total length.
    paths = [([(0, 0)], m[0,0])]

    # Loops while there are uncompleted paths.
    while paths:

        # Unpacking the longest path
        short_path, short_length = paths.pop()

        last_node = short_path[-1]

        #print(short_length, short_path)


        # Success condition
        if last_node == (79, 79):
            print(short_length, short_path)
            return short_path, short_length


        # Creating two new paths by expanding to the next two nodes

        if last_node[0] < m.shape[0] - 1:
            down_node = tuple(map(lambda x, y: x + y, last_node, (1, 0)))
            down_length = short_length + m[down_node]

        if last_node[1] < m.shape[1] - 1:
            right_node = tuple(map(lambda x, y: x + y, last_node, (0, 1)))
            right_length  = short_length + m[right_node]


        # Comparing length of new paths for inserting them in the list of paths
        if down_length < right_length:
            new_short_path = (short_path + [down_node], down_length)
            new_long_path  = (short_path + [right_node], right_length)
        else:
            new_short_path = (short_path + [right_node], right_length)
            new_long_path  = (short_path + [down_node], down_length)


        SHORT_PATH_INSERTED = False
        LONG_PATH_INSERTED  = False
        i = len(paths) - 1
        while i >= 0:
            # if two paths visit the same node only the shortest path is kept
            if new_short_path[0][-1] == paths[i][0][-1]:
                if new_short_path[1] < paths[i][1]:
                    paths.pop(i)
                else:
                    SHORT_PATH_INSERTED = True

            if new_long_path[0][-1] == paths[i][0][-1]:
                if new_long_path[1] < paths[i][1]:
                    paths.pop(i)
                else:
                    LONG_PATH_INSERTED = True

            # the paths are inserted in diminishing order of length
            if not SHORT_PATH_INSERTED and paths[i][1] > new_short_path[1]:
                paths.insert(i+1, new_short_path)
                SHORT_PATH_INSERTED = True

            if not LONG_PATH_INSERTED and paths[i][1] > new_long_path[1]:
                paths.insert(i+1, new_long_path)
                LONG_PATH_INSERTED = True

            i -= 1

        # if no paths are longer, the new paths are inserted at the beginning
        if not SHORT_PATH_INSERTED:
            paths.insert(0, new_short_path)
        if not LONG_PATH_INSERTED:
            paths.insert(0, new_long_path)

def triangle(n):
    return n * (n + 1) // 2

def pentagonal(n):
    return n * (3*n - 1) // 2

def hexagonal(n):
    return n * (2 * n - 1)

def heptagonal(n):
    return n * (5 * n - 3) // 2

def octagonal(n):
    return n * (3 * n - 2)

def create_polygonal_numbers():
    x3 = []
    i = 30
    x = triangle(i)
    while x < 9999:
        if x % 100 < 10:
            pass
        elif x >= 1000:
            x3.append(x)
        i += 1
        x = triangle(i)

    x4 = []
    i = 30
    x = i**2
    while x < 9999:
        if x % 100 < 10:
            pass
        elif x >= 1000:
            x4.append(x)
        i += 1
        x = i**2

    x5 = []
    i = 10
    x = pentagonal(i)
    while x < 9999:
        if x % 100 < 10:
            pass
        elif x >= 1000:
            x5.append(x)
        i += 1
        x = pentagonal(i)

    x6 = []
    i = 9
    x = hexagonal(i)
    while x < 9999:
        if x % 100 < 10:
            pass
        elif x >= 1000:
            x6.append(x)
        i += 1
        x = hexagonal(i)

    x7 = []
    i = 8
    x = heptagonal(i)
    while x < 9999:
        if x % 100 < 10:
            pass
        elif x >= 1000:
            x7.append(x)
        i += 1
        x = heptagonal(i)

    x8 = []
    i = 8
    x = octagonal(i)
    while x < 9999:
        if x % 100 < 10:
            pass
        elif x >= 1000:
            x8.append(x)
        i += 1
        x = octagonal(i)

    return [x3,x4,x5,x6,x7,x8]

def cyclical_numbers(number_sets, chain=None):
    #print(chain, len(number_sets))
    if chain is None:
        nums = number_sets[-1]
        for n in nums:
            new_chain = cyclical_numbers(number_sets[:-1], [n])
            if len(new_chain) == 6:
                return new_chain
    elif len(chain) == 5:
        x = int(str(chain[-1])[-2:] + str(chain[0])[:2])
        #print(x, number_sets)
        if x in number_sets[0]:
            return chain + [x]
    else:
        x = int(str(chain[-1])[-2:])
        for i in range(len(number_sets)):
            for n in number_sets[i]:
                if n // 100 == x:
                    new_chain = cyclical_numbers(number_sets[:i] + number_sets[i+1:], chain + [n])
                    if len(new_chain) == 6:
                        return new_chain
    return chain

def OP(sequence):
    k = len(sequence)
    A = np.zeros((k,k), dtype=int)
    for i in range(k):
        for j in range(k):
            A[i,j] = (i+1)**j
    solution = np.linalg.solve(A, np.array(sequence))
    return lambda n: round(sum([solution[i]*n**i for i in range(k)]))

def optimumPoly():
    u = lambda n: 1 - n + n**2 - n**3 + n**4 - n**5 + n**6 - n**7 + n**8 - n**9 + n**10
    sequence = [u(n) for n in range(1,14)]


    FIT = 0
    for k in range(1,13):
        v = OP(sequence[:k])
        if v(k+1) != sequence[k]:
            print(k, v(k+1), sequence[k])
            FIT += v(k+1)
    print(FIT)

def pell_equation(D):
    """Calculates the minimal solution for x and y for the equation:
       x**2 - D * y**2 = 1"""
    P0 = 0
    Q0 = 1
    a0 = int(sqrt(D))
    p0 = a0
    q0 = 1

    P1 = a0
    Q1 = D - a0**2
    a1 = (a0 + P1) // Q1
    p1 = a0*a1 + 1
    q1 = a1

    r = 1
    while True:
        P1, P0 = a1*Q1 - P1, P1
        Q1, Q0 = (D - P1**2) // Q1, Q1
        a1 = (a0 + P1) // Q1
        p1, p0 = a1*p1 + p0, p1
        q1, q0 = a1*q1 + q0, q1
        r += 1
        if a1 == 2*a0:
            break

    if r % 2 != 0:
        for k in range(r):
            P1, P0 = a1*Q1 - P1, P1
            Q1, Q0 = (D - P1**2) // Q1, Q1
            a1 = (a0 + P1) // Q1
            p1, p0 = a1*p1 + p0, p1
            q1, q0 = a1*q1 + q0, q1
            r += 1

    return p0, q0

def digit_factorial(x):
    """Returns the sum of the factorials of the digits of the number"""
    return np.sum([factorial(int(d)) for d in str(x)])

def factorial_chain():
    """Counts the number of digit factorial chains of length 60 with a starting number under one million"""
    len60_chains = 0
    chain_lengths = np.zeros(2177281, dtype=int)
    chain_lengths[np.array([0, 1, 2, 145, 40585])] = 1
    chain_lengths[np.array([871, 872, 45361, 45362])] = 2
    chain_lengths[np.array([169, 1454, 363601])] = 3
    for n in range(3, 1000001):
        chain = []
        chain_len = 0
        while chain_lengths[n] == 0:
            chain.append(n)
            chain_len += 1
            n = digit_factorial(n)
        chain_len += chain_lengths[n]

        if chain_len == 60:
            len60_chains += 1

        for k in chain:
            chain_lengths[k] = chain_len
            chain_len -= 1
    print(len60_chains)

def digit_square(x):
    """Returns the sum of the square of the digits of the number"""
    return np.sum([int(d)**2 for d in str(x)])

def square_chain(N=10000000):
    """Counts the number of digit square chains converging to 89 with a starting number under N"""
    num_89_chains = 1
    checked = np.zeros(N+1, dtype=int)
    checked[1] = 1
    checked[89] = 89
    for n in range(1, N+1):
        print(n)
        if checked[n] == 0:
            chain = [n]
            n = digit_square(n)
            while checked[n] == 0:
                chain.append(n)
                n = digit_square(n)
            chain = np.array(chain)
            if checked[n] == 1:
                checked[chain] = 1
            elif checked[n] == 89:
                checked[chain] = 89
                num_89_chains += len(chain)
    print('Number of chains under {} converging to 89: {}'.format(N, num_89_chains))

def triangle_containment():
    T = np.genfromtxt('triangles', delimiter=',')
    X = T[:,::2]
    Y = T[:,1::2]
    A = np.arctan(Y / X) * 180 / np.pi
    A = np.where(np.logical_and(X < 0, Y > 0), A + 180, A)
    A = np.where(np.logical_and(X < 0, Y < 0), A - 180, A)
    A.sort(axis=1)
    Amin = A[:,0]
    Amed = A[:,1]
    Amax = A[:,2]
    print(Amin[0], Amed[0], Amax[0])
    print(Amax[0] - 180, Amed[0], Amin[0] + 180)
    X = np.logical_and(Amed > Amax - 180, Amed < Amin + 180)
    print(np.sum(X))
    #for a, b, c in A:
    #    print('{: 6.2f}  {: 6.2f}  {: 6.2f}'.format(a, b, c))

def plot_triangle():
    T = np.genfromtxt('triangles', delimiter=',')
    X = T[:,::2]
    Y = T[:,1::2]
    A = np.arctan(Y / X) * 180 / np.pi
    A = np.where(np.logical_and(X < 0, Y > 0), A + 180, A)
    A = np.where(np.logical_and(X < 0, Y < 0), A - 180, A)
    A.sort(axis=1)
    Amin = A[:,0]
    Amed = A[:,1]
    Amax = A[:,2]
    print(np.any(Amax - Amed == 180))
    print(np.any(Amed - Amin == 180))
    print(np.any(Amed - Amin == 180))
    I = 1*np.logical_and.reduce([Amax - Amin >= 180,Amed >= Amax - 180, Amed <= Amin + 180])
    I = 1*np.logical_and.reduce([Amax - Amin >= 180,Amax - Amed <= 180, Amed - Amin <= 180])
    I = 1*I
    print(np.sum(I))

    col = ['r','g']
    n = 20
    #index = np.random.randint(0, 1000, n)
    """index = np.arange(0, 1000)
    for i in index:
        plt.clf()
        plt.plot(0, 0, 'bo')
        plt.xlim([-1000, 1000])
        plt.ylim([-1000, 1000])
        plt.plot(X[i], Y[i], c=col[I[i]])
        plt.plot((X[i,2], X[i,0]), (Y[i,2], Y[i,0]), c=col[I[i]])
        plt.pause(0.2)
    plt.show()"""

def paper_distribution(num_papers, total_papers):
    x = npr.uniform(0,total_papers, total_papers.size)
    new_papers = np.zeros_like(num_papers)
    threshold = np.copy(num_papers[:, 0])
    new_papers[:,0] = np.where(x < threshold, -1, 0)
    for k in range(1, 4):
        threshold += num_papers[:, k]
        new_papers[:,k] = np.where(new_papers[:,k-1] != 0, 1, np.where(x < threshold, -1, 0))
    return new_papers

def paper_sheets_batch(batch_size=500000, epochs=500):
    single_papers = 0
    weeks = 0
    single_papers = 232216232 # 0.464432
    weeks = 500000000
    for e in range(epochs):
        num_papers = np.ones((batch_size, 4), dtype=int)
        for job in range(14):

            total_papers = np.sum(num_papers, axis=1)
            single_papers += np.sum(total_papers == 1)

            new_papers = paper_distribution(num_papers, total_papers)
            num_papers = num_papers + new_papers

        weeks += batch_size
        print(e, single_papers, weeks, round(single_papers / weeks, 6))
    print(single_papers, weeks)
    print(single_papers / weeks)

def paper(P=np.ones(4, dtype=int), n=14):
    #print((14-n)*'| ',P)
    if n == 0:
        return 0

    tot_P = np.sum(P)
    single_papers = 0
    if tot_P == 1:
        single_papers += 1

    for i in range(4):
        if P[i] == 0:
            continue

        newP = copy(P)
        newP[i] -= 1
        newP[i+1:] += 1

        proba = paper(newP, n-1)
        #if proba != 0: print(proba)
        single_papers += P[i] * proba / tot_P
    #if single_papers != 0: print(single_papers)
    return single_papers

def prime_sieve2(N):
    P = np.ones(N+1, dtype=bool)
    P[0] = P[1] = False
    for n in range(2, int(sqrt(N))):
        p = P[n]
        if p:
            P[2*n::n] = False
    return P

def prime_gen_int(N):
    isPrime = prime_sieve2(N+1)
    PGI = np.ones(N+1, dtype=bool)
    PGI[0] = False
    for n in range(1, int(sqrt(N))):
        a = PGI[n::n]
        b = isPrime[n+1: n+1+len(a)]
        PGI[n::n] = a & b

    K = np.arange(N+1)[PGI]
    print(K)
    print(np.sum(K))

def fully_connected(G):
    n = len(G)
    visited = np.zeros(n, dtype=bool)

    def fill(G, node=0):
        visited[node] = True
        for j in range(n):
            if G[node, j] and not visited[j]:
                fill(G, j)

    fill(G)
    return visited.all()


def kruskal(G):
    n = len(G)
    reducedG = np.zeros_like(G)
    forest = np.arange(n)
    mask = np.ones_like(G)
    while (forest != 0).any() and (mask*G).any():
        maskG = mask*G
        i, j = np.where(maskG == np.min(np.ma.masked_equal(maskG, 0)))
        i, j = i[0], j[0]
        mask[i, j] = 0
        mask[j, i] = 0
        a = forest[i]
        b = forest[j]
        if a != b:
            reducedG[i, j] = G[i,j]
            reducedG[j, i] = G[i,j]
            forest[np.where(forest == b)] = a
    return reducedG

def reduce_graph_elimination():
    #M = np.genfromtxt('test_network', dtype=int, delimiter=',', filling_values=0)
    M = np.genfromtxt('network', dtype=int, delimiter=',', filling_values=0)
    reducedM = np.copy(M)
    n = len(M)
    nonZero = np.sum(M > 0)
    mask1 = np.ones_like(M)
    while nonZero > n - 1:
        maxIndex = np.where(M == np.max(mask1*M))
        mask1[maxIndex] = 0
        mask2 = np.ones_like(M)
        mask2[maxIndex] = 0
        if fully_connected(mask2*reducedM):
            reducedM[maxIndex] = 0
            nonZero -= 1
    return reducedM

def attraction(x, y, M, a, r):
    n = len(M)
    l = 1.
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    D = np.zeros((n,n))
    for i in range(n):
        D[i, i+1:] = np.sqrt((x[i] - x[i+1:])**2 + (y[i] - y[i+1:])**2)
        D[i+1:, i] = D[i, i+1:]
        for j in range(n):
            if i == j:
                continue
            dx[i] = dx[i] - r * (x[j] - x[i]) / D[i,j]**2
            dy[i] = dy[i] - r * (y[j] - y[i]) / D[i,j]**2

            if M[i, j] and D[i,j] > l:
                dx[i] = dx[i] + a * (x[j] - x[i]) * D[i,j]
                dy[i] = dy[i] + a * (y[j] - y[i]) * D[i,j]
    return dx, dy

def draw_graph(M, a=1., r=1., labels=None):
    n = len(M)
    if labels is None:
        labels = np.arange(1, n+1, dtype=int)
    s = np.linspace(0, (1-1/n)*2*np.pi, n)
    x = n*np.cos(s)
    y = n*np.sin(s)

    dt = 0.001
    t = 0
    T = 10
    while t < T:
        dx, dy = attraction(x, y, M, a, r)
        x = x + dt*dx
        y = y + dt*dy
        t += dt

    for i in range(n):
        plt.plot(x[i], y[i], 'bo')
        plt.annotate(labels[i], (x[i], y[i]), fontsize=12, color='red')
        for j in range(i+1, n):
            if M[i, j]:
                plt.plot((x[i], x[j]),(y[i], y[j]), 'b')
                textx = (x[i] + x[j]) / 2
                texty = (y[i] + y[j]) / 2
                plt.annotate(M[i,j], (textx, texty))
    plt.show()

def draw_graph2(M, a=.5, r=1., labels=None, eps=0.1):
    n = len(M)
    if labels is None:
        labels = np.arange(1, n+1, dtype=int)
    s = np.linspace(0, (1-1/n)*2*np.pi, n)
    x = n * np.cos(s)
    y = n * np.sin(s)
    P = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    for i in range(n):
        P.append([ax.plot(x[i], y[i], 'bo')[0]])
        for j in range(i+1, n):
            if M[i, j]:
                P[-1].append(ax.plot((x[i], x[j]),(y[i], y[j]), 'b', lw=0.1)[0])
            else:
                P[-1].append(None)
    plt.pause(2)
    #plt.close()
    dt = 0.01
    t = 0
    T = 5
    while t < T:
        dx, dy = attraction(x, y, M, a, r)
        if dt*np.max(dx) < eps and dt*np.max(dy) < eps:
            break
        x = x + dt*dx
        y = y + dt*dy
        t += dt

        for i in range(n):
            P[i][0].set_data(x[i], y[i])
            for j in range(i + 1, n):
                if M[i, j]:
                    P[i][j - i].set_data((x[i], x[j]),(y[i], y[j]))
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.0001)


    for i in range(n):
        #plt.plot(x[i], y[i], 'bo')
        plt.annotate(labels[i], (x[i], y[i]), fontsize=8, color='red')
        if n < 50:
            for j in range(i+1, n):
                if M[i, j]:
                    #plt.plot((x[i], x[j]),(y[i], y[j]), 'b')
                    textx = (x[i] + x[j]) / 2
                    texty = (y[i] + y[j]) / 2
                    plt.annotate(M[i,j], (textx, texty), fontsize=6)
    plt.pause(1)
    plt.show()

def graphLoopRemove(M, first=0):
    """Returns a copy of graph G where in each loop the edge with the largest weight has been removed"""
    G = np.copy(M)
    n = len(G)

    def removeLoops(path):
        i = path[-1]
        for j in range(n):
            if G[i,j] != 0 and (len(path) == 1 or j != path[-2]):
                if j not in path:
                    removeLoops(path + [j])
                else:
                    loop = path[path.index(j):]
                    loopLen = len(loop)
                    loopBroken = False
                    maxk = 0
                    maxa, maxb = 0, 1
                    for k in range(1, loopLen):
                        a = loop[k]
                        b = loop[(k + 1) % loopLen]
                        if G[a, b] == 0:
                            loopBroken = True
                            break
                        else:
                            if G[a, b] > G[maxa, maxb]:
                                maxa, maxb = a, b

                    if not loopBroken:
                        G[maxa, maxb] = 0
                        G[maxb, maxa] = 0

    removeLoops([first])
    return G

def semiprimes():
    N = 100000000
    P = prime_sieve(N // 2)
    total = 0
    for i in range(len(P)):
        semip = np.sum((P[i]*P[i:]) <= N)
        total += semip
        if semip == 0:
            break
    print(total)

def dice_game():
    P = np.zeros((37,9))
    P[0] = 1
    C = np.zeros((37,6))

    def calcP(n, dice=9):
        if n < 0:
            return 1

        if dice == 0:
            return 0

        if P[n, dice-1]:
            return P[n, dice-1]

        proba = 0.
        for k in range(1, 5):
            proba += calcP(n - k, dice - 1)

        proba /= 4
        P[n, dice-1] = proba
        return proba

    def calcC(n, dice=6):
        if n < 0:
            return 0

        if dice == 0:
            return int(n == 0)

        if C[n, dice-1]:
            return C[n, dice-1]

        proba = 0.
        for k in range(1, 7):
            proba += calcC(n - k, dice - 1)

        proba /= 6
        C[n, dice-1] = proba
        return proba


    for n in range(1, 37):
        calcP(n)

    proba = 0.
    for n in range(1, 37):
        calcC(n)
        proba += C[n,-1] * P[n,-1]

    print(round(proba, 7))

def number_splitting():
    N = 1000000
    num = np.arange(2, N+1)
    num2 = num**2

    def sumPossible(n, n2):
        if n2 < 10:
            return n == n2
        if n2 < n:
            return False

        if n == n2:
            return True

        k = 10
        while n2 >= k:
            d, m = divmod(n2, k)
            result = sumPossible(n - m, d)
            if result:
                return True
            k *= 10

        return result

    S = 0
    for n, n2 in zip(num, num2):
        m1 = digit_sum(n) % 9
        m2 = digit_sum(n2) % 9
        if m1 != m2:
            continue
        if sumPossible(n, n2):
            print(n, n2)
            S += n2
    print(S)

def powers_of_two(L, n):
    log10of2 = log10(2)
    j = 90
    prev_j = 0
    k = 1
    l = len(L)-1
    while k < n:
        for next_j in [j+196, j+289, j+485]:
            frac, power = modf(next_j * log10of2)
            s = str(pow(10,frac)*100)
            if s[:3] == L:
                j = next_j
                k += 1
                break
    return j


def sumEulercoins():
    a = 1504170715041707 # 17, 1249, 12043, 5882353
    b = 4503599627370517
    count = a
    prev_eulercoin = a
    eulercoin = (3*a) % b
    print(prev_eulercoin)
    print(eulercoin)
    print((prev_eulercoin - eulercoin) % eulercoin)
    while eulercoin > 1:
        count += eulercoin
        eulercoin, prev_eulercoin = eulercoin - (prev_eulercoin - eulercoin) % eulercoin, eulercoin
        print('{:>17} {} {}'.format(eulercoin, prev_eulercoin - eulercoin, count))

    print(count)


def integer_triangles():
    Lmax = 120
    py_triangles = np.zeros(Lmax+1, dtype=int)
    vmax = int((sqrt(2*Lmax+1)-1)/2)
    for v in range(1, vmax+1):
        umax = int((Lmax - 2*v**2) / (2*v))
        for u in range(1, min(v, umax+1)):
            if gcd(u,v) != 1 or (u % 2 == v % 2): continue
            L = 2*v**2 + 2*u*v
            a = v**2 - u**2
            b = 2*u*v
            c = v**2+u**2
            print(a, b, c, L)
            py_triangles[L::L] += 1

    print(np.sum(np.where(py_triangles == 1, 1, 0)))

def countDuboidSolutions(M):
    count = 0
    vmax = M // 2
    for v in range(1, vmax + 1):
        for u in range(1, v):
            if gcd(u, v) != 1 or (u % 2 == v % 2): continue

            x = v**2 - u**2
            y = 2*u*v
            a, b = max(x, y), min(x, y)
            if b > M or a > 2*M: continue

            sr = v**2+u**2
            for k in range(1, M // a + 1):
                count += k*b // 2 + max(0, (k*a)//2 - k*a + k*b + 1)
            for k in range(M // a + 1, M // b + 1):
                count += max(0, (k * a) // 2 - k * a + k * b + 1)
    return count

def cuboid_route():
    maxSolutions = 0
    Mmin = 200
    Mmax = 2000
    M = 0
    while M != (Mmin + Mmax) // 2:
        M = (Mmin + Mmax) // 2
        count = countDuboidSolutions(M)
        print(M, count)
        if count > 1000000:
            Mmax = M
        else:
            Mmin = M

    M += 1
    print(M, countDuboidSolutions(M))
    return M

def rank_hand(hand):
    hand = sorted(hand, key=lambda x: x['val'])

    flush = True
    straight = True
    same_val = 1
    high_same_val = 1
    high_val = hand[0]['val']
    second_high_same_val = 0
    second_val = 0
    for i in range(4):
        if hand[i+1]['suite'] != hand[i]['suite']: # Check Flush
            flush = False

        if hand[i+1]['val'] != hand[i]['val'] + 1:
            straight = False

        if hand[i+1]['val'] == hand[i]['val']:
            same_val += 1
        else:
            if same_val >= high_same_val:
                second_high_same_val = high_same_val
                second_val = high_val

                high_same_val = same_val
                high_val = hand[i]['val']
            else:
                if same_val >= second_high_same_val:
                    second_high_same_val = same_val
                    second_val = hand[i]['val']

            same_val = 1

    if same_val >= high_same_val:
        second_high_same_val = high_same_val
        second_val = high_val

        high_same_val = same_val
        high_val = hand[i]['val']
    else:
        if same_val >= second_high_same_val:
            second_high_same_val = same_val
            second_val = hand[i]['val']

    if flush and straight:
        return 8, hand[-1]['val'], hand[-1]['val']

    if high_same_val == 4:
        return 7, high_val, hand[-1]['val']

    if high_same_val == 3 and second_high_same_val == 2:
        return 6, (high_val, second_val), hand[-1]['val']

    if flush:
        return 5, hand[-1]['val'], hand[-1]['val']

    if straight:
        return 4, hand[-1]['val'], hand[-1]['val']

    if high_same_val == 3:
        return 3, high_val, hand[-1]['val']

    if high_same_val == 2 and second_high_same_val == 2:
        return 2, (high_val, second_val), hand[-1]['val']

    if high_same_val == 2:
        return 1, high_val, hand[-1]['val']

    return 0, hand[-1]['val'], hand[-1]['val']

def poker_hands():
    player1wins = 0
    f = open('poker')
    valtoint = dict()
    for k in range(2, 10):
        valtoint[str(k)] = k
    valtoint['T'] = 10
    valtoint['J'] = 11
    valtoint['Q'] = 12
    valtoint['K'] = 13
    valtoint['A'] = 14

    for line in f:
        p1 = []
        p2 = []
        for i in range(0, 13, 3):
            val = valtoint[line[i]]
            suite = line[i+1]
            p1.append({'val':val, 'suite':suite})

            val = valtoint[line[i+15]]
            suite = line[i+16]
            p2.append({'val':val, 'suite':suite})

        rank1, combo_val1, high_card1 = rank_hand(p1)
        rank2, combo_val2, high_card2 = rank_hand(p2)

        print('P1 rank: {} {:>8}, P2 rank: {} {:>8}'.format(rank1, str(combo_val1), rank2, str(combo_val2)))

        if rank1 > rank2:
            player1wins += 1
        elif rank1 == rank2:
            if rank1 in (2,6):
                if combo_val1[0] > combo_val2[0]:
                    player1wins += 1
                elif combo_val1[0] == combo_val2[0]:
                    if combo_val1[1] > combo_val2[1]:
                        player1wins += 1
                    elif combo_val1[1] == combo_val2[1]:
                        if high_card1 > high_card2:
                            player1wins += 1

            if combo_val1 > combo_val2:
                player1wins += 1
            elif combo_val1 == combo_val2:
                if high_card1 > high_card2:
                    player1wins += 1

    print(player1wins)



    f.close()

def base_convert(n, newbase):
    x = 0
    while n > 0:
        a = int(log(n, newbase))
        c = newbase**a
        b = n // c
        x += b * 10 ** a
        n -= b * c
    return x

def pascal2(n):
    if n < 8:
        return (n * (n + 1)) // 2

    a = int(log(n, 7))
    c = 7 ** a
    b = n // c

    return (b * (b+1)) // 2 * 28**a + (b+1) * pascal2(n - b*c)
