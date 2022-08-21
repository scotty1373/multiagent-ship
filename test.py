# -*-  coding=utf-8 -*-
# @Time : 2022/8/18 16:35
# @Author : Scotty1373
# @File : test.py
# @Software : PyCharm
std_in = '5 cats dog sand and cat catsandog'

max_lens = 4
res_lens = 3
iss_score = [445, 754, 553, 122]
iss_prob = [89, 38, 76, 23]

def max_scores(max_lens, res_lens, iss_score, iss_prob):
    # 创建dp矩阵
    dp = [0 for i in range(max_lens)]

    mapping = []
    for i in range(len(iss_score)):
        mapping.append([iss_score[i], iss_prob[i]])

    # 按iss_score排序，并映射到iss_prob上
    sorted_id = mapping.sort(key=lambda x: x[0], reverse=True)

    counter = 0
    while counter < max_lens:
        if not counter:
            dp[counter] = mapping[counter][0]
        elif counter < res_lens:
            dp[counter] = mapping[counter][0] + dp[counter - 1]
        else:
            dp[counter] = mapping[counter][0] * mapping[counter][1] / 100 + dp[counter - 1]
        counter += 1

    print(f'{dp[-1]:.2f}')

sub_str = 'redrde'

def getSubstringWithEqual012(string):
    N = len(string)

    # map to store, how many times a difference
    # pair has occurred previously
    mp = dict()
    mp[(0, 0)] = 1

    # zc (Count of zeroes), oc(Count of 1s)
    # and tc(count of twos)
    # In starting all counts are zero
    zc, oc, tc = 0, 0, 0

    # looping into string
    res = 0  # Initialize result
    for i in range(N):

        # increasing the count of current character
        if string[i] == 'r':
            zc += 1
        elif string[i] == 'e':
            oc += 1
        else:
            tc += 1  # Assuming that string doesn't contain
            # other characters

        # making pair of differences (z[i] - o[i],
        # z[i] - t[i])
        tmp = (zc - oc, zc - tc)

        # Count of previous occurrences of above pair
        # indicates that the subarrays forming from
        # every previous occurrence to this occurrence
        # is a subarray with equal number of 0's, 1's
        # and 2's
        if tmp not in mp:
            res += 0
        else:
            res += mp[tmp]

        # increasing the count of current difference
        # pair by 1
        if tmp in mp:
            mp[tmp] += 1
        else:
            mp[tmp] = 1

    return res


max_lt_lens = 5
select_lens = 2
lt = [2, 3, 4, 5, 6]

def subarrayBitwiseANDs(A, max, sele):
    dp = [0 for i in range(10)]
    offset = 1
    for idx in range(max):
        for b in A:
            dp[idx] += 1 if b & offset else 0
        offset <<= 1
    dp.reverse()


if __name__ == '__main__':
    # max_scores(max_lens, res_lens, iss_score, iss_prob)
    print(getSubstringWithEqual012(sub_str))
    # subarrayBitwiseANDs(lt, max_lt_lens, select_lens)
