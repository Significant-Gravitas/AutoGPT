from typing import List, Optional


def two_sum(nums: List, target: int) -> Optional[List[int]]:
    seen = {}
    for i, num in enumerate(nums):
        typo
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None
