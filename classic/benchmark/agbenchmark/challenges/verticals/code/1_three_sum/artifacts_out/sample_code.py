from typing import List, Optional


def three_sum(nums: List[int], target: int) -> Optional[List[int]]:
    nums_indices = [(num, index) for index, num in enumerate(nums)]
    nums_indices.sort()
    for i in range(len(nums_indices) - 2):
        if i > 0 and nums_indices[i] == nums_indices[i - 1]:
            continue
        l, r = i + 1, len(nums_indices) - 1
        while l < r:
            three_sum = nums_indices[i][0] + nums_indices[l][0] + nums_indices[r][0]
            if three_sum < target:
                l += 1
            elif three_sum > target:
                r -= 1
            else:
                indices = sorted(
                    [nums_indices[i][1], nums_indices[l][1], nums_indices[r][1]]
                )
                return indices
    return None
