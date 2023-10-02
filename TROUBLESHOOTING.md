This page is a list of issues you could encounter along with their fixes.

# Forge
**Poetry configuration invalid**

The poetry configuration is invalid: 
- Additional properties are not allowed ('group' was unexpected)
<img width="487" alt="Screenshot 2023-09-22 at 5 42 59 PM" src="https://github.com/Significant-Gravitas/AutoGPT/assets/9652976/dd451e6b-8114-44de-9928-075f5f06d661">

**Pydantic Validation Error**

Remove your sqlite agent.db file. it's probably because some of your data is not complying with the new spec (we will create migrations soon to avoid this problem)


*Solution*

Update poetry

# Benchmark
TODO

# Frontend
TODO
