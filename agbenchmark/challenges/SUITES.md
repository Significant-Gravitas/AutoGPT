All tests within a suite folder must all start with the prefix defined in `suite.json`. There are two types of suites.

#### same_task

If same_task is set to true, all of the data.jsons are combined into one test. A single test runs, but multiple regression tests, internal_infos, dependencies, and reports are created. The artifacts_in/out and custom python should be in the suite folder as it's shared between tests. **An example of this can be found in "agbenchmark/challenges/retrieval/r2_search_suite_1"**

```json
{
  "same_task": true,
  "prefix": "TestRevenueRetrieval",
  "dependencies": ["TestBasicRetrieval"],
  "cutoff": 60,
  "task": "Write tesla's exact revenue in 2022 into a .txt file. Use the US notation, with a precision rounded to the nearest million dollars (for instance, $31,578 billion).",
  "shared_category": ["retrieval"]
}
```

The structure for a same_task report looks like this:

```
"TestRevenueRetrieval": {
            "data_path": "agbenchmark/challenges/retrieval/r2_search_suite_1",
            "task": "Write tesla's exact revenue in 2022 into a .txt file. Use the US notation, with a precision rounded to the nearest million dollars (for instance, $31,578 billion).",
            "category": [
                "retrieval"
            ],
            "metrics": {
                "percentage": 100.0,
                "highest_difficulty": "intermediate",
                "run_time": "0.016 seconds"
            },
            "tests": {
                "TestRevenueRetrieval_1.0": {
                    "data_path": "agbenchmark/challenges/retrieval/r2_search_suite_1/1_tesla_revenue/data.json",
                    "is_regression": false,
                    "answer": "It was $81.462 billion in 2022.",
                    "description": "A no guardrails search for info",
                    "metrics": {
                        "difficulty": "novice",
                        "success": true,
                        "success_%": 100.0
                    }
                },
                "TestRevenueRetrieval_1.1": {
                    "data_path": "agbenchmark/challenges/retrieval/r2_search_suite_1/2_specific/data.json",
                    "is_regression": false,
                    "answer": "It was $81.462 billion in 2022.",
                    "description": "This one checks the accuracy of the information over r2",
                    "metrics": {
                        "difficulty": "novice",
                        "success": true,
                        "success_%": 0
                    }
                },
            },
            "reached_cutoff": false
        },
```

#### same_task

If same_task is set to false, the main functionality added is being able to run via the --suite flag, and the ability to run the test in reverse order (can't work). Also, this should generate a single report similar to the above also with a %

```json
{
  "same_task": false,
  "reverse_order": true,
  "prefix": "TestReturnCode"
}
```

The structure for a non same_task report looks like this:

```
"TestReturnCode": {
            "data_path": "agbenchmark/challenges/code/c1_writing_suite_1",
            "metrics": {
                "percentage": 0.0,
                "highest_difficulty": "No successful tests",
                "run_time": "15.972 seconds"
            },
            "tests": {
                "TestReturnCode_Simple": {
                    "data_path": "agbenchmark/challenges/code/c1_writing_suite_1/1_return/data.json",
                    "is_regression": false,
                    "category": [
                        "code",
                        "iterate"
                    ],
                    "task": "Return the multiplied number in the function multiply_int in code.py. You can make sure you have correctly done this by running test.py",
                    "answer": "Just a simple multiple by 2 function. Num is 4 so answer is 8",
                    "description": "Simple test if a simple code instruction can be executed",
                    "metrics": {
                        "difficulty": "basic",
                        "success": false,
                        "fail_reason": "assert 1 in [0.0]",
                        "success_%": 0.0,
                        "run_time": "15.96 seconds"
                    },
                    "reached_cutoff": false
                },
                "TestReturnCode_Write": {
                    "data_path": "agbenchmark/challenges/code/c1_writing_suite_1/2_write/data.json",
                    "is_regression": false,
                    "category": [
                        "code",
                        "iterate"
                    ],
                    "task": "Add a function called multiply_int in code.py that multiplies numbers by 2. You can make sure you have correctly done this by running test.py",
                    "answer": "Just a simple multiple by 2 function. Num is 4 so answer is 8",
                    "description": "Small step up, just writing the function with a name as well as the return statement.",
                    "metrics": {
                        "difficulty": "novice",
                        "success": false,
                        "fail_reason": "agbenchmark/challenges/test_all.py::TestReturnCode_Write::test_method[challenge_data0] depends on agbenchmark/challenges/test_all.py::TestReturnCode_Simple::test_method[challenge_data0]",
                        "success_%": 0.0,
                        "run_time": "0.004 seconds"
                    },
                    "reached_cutoff": false
                },
            }
        }
```
