import itertools

# iterated weights share of the inheritance item list
inheritance = [8, 0.5, 3.5, 6, 1.2, 1.2, 1.2, 0.3, 0.6, 1, 2, 1]

# the total inheritance value
total_inherit = sum(inheritance)

# the target inheritance plan for each daughter
target_inherit = total_inherit / 3

# indexing the inheritance list
indices = list(range(len(inheritance)))

# initialize the minimum difference and the optimal partition
min_diff = float('inf')
opt_partition = None

# all possible combinations should be generated, considering the inheritance
# the given item list is not too long, the brute-force method is acceptable


def find_partitions():
    global min_diff, opt_partition
    for inheritance1_item in range(1, len(inheritance) - 1):
        for inheritance1_indices in itertools.combinations(indices, inheritance1_item):
            inheritance1 = [inheritance[i] for i in inheritance1_indices]
            sum_inheritance1 = sum(inheritance1)
            if abs(sum_inheritance1 - target_inherit) > min_diff:
                continue
            remaining_indices1 = list(set(indices) - set(inheritance1_indices))
            for inheritance2_size in range(1, len(remaining_indices1)):
                for inheritance2_indices in itertools.combinations(remaining_indices1, inheritance2_size):
                    inheritance2 = [inheritance[i] for i in inheritance2_indices]
                    sum_inheritance2 = sum(inheritance2)
                    inheritance3_indices = list(set(remaining_indices1) - set(inheritance2_indices))
                    inheritance3 = [inheritance[i] for i in inheritance3_indices]
                    sum_inheritance3 = sum(inheritance3)
                    sums_inheritance = [sum_inheritance1, sum_inheritance2, sum_inheritance3]
                    max_diff = max(sums_inheritance) - min(sums_inheritance)
                    if max_diff < min_diff:
                        min_diff = max_diff
                        opt_partition = (inheritance1, inheritance2, inheritance3)
                        if min_diff <= 0.1:
                            return


# output the result
find_partitions()

if opt_partition:
    inheritance1, inheritance2, inheritance3 = opt_partition
    print("the first share of the inheritance is：", inheritance1, "total value as：", sum(inheritance1))
    print("the second share of the inheritance is：", inheritance2, "total value as：", sum(inheritance2))
    print("the third share of the inheritance is：", inheritance3, "total value as：", sum(inheritance3))
    print("maximum difference between the inheritance partitions：", min_diff)
else:
    print("no optimal partition found")
