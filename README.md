# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Significant-Gravitas/Auto-GPT/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| autogpt/\_\_init\_\_.py                     |        0 |        0 |        0 |        0 |    100% |           |
| autogpt/\_\_main\_\_.py                     |        3 |        3 |        2 |        0 |      0% |       2-5 |
| autogpt/agent/\_\_init\_\_.py               |        3 |        0 |        0 |        0 |    100% |           |
| autogpt/agent/agent.py                      |      102 |       89 |       44 |        0 |      9% |54-61, 65-226 |
| autogpt/agent/agent\_manager.py             |       69 |       55 |       40 |        0 |     13% |33-71, 83-119, 129, 141-145 |
| autogpt/app.py                              |       98 |       69 |       36 |        0 |     22% |27-31, 48-72, 79-87, 105-135, 151-153, 166, 171-172, 193-209, 216-224, 234, 249-250 |
| autogpt/chat.py                             |       70 |       53 |       20 |        0 |     19% |    59-195 |
| autogpt/cli.py                              |       74 |       74 |       14 |        0 |      0% |     2-181 |
| autogpt/commands/\_\_init\_\_.py            |        0 |        0 |        0 |        0 |    100% |           |
| autogpt/commands/analyze\_code.py           |        9 |        9 |        0 |        0 |      0% |      2-31 |
| autogpt/commands/audio\_text.py             |       22 |       22 |        4 |        0 |      0% |      2-63 |
| autogpt/commands/command.py                 |       66 |       12 |       18 |        3 |     77% |36, 55, 58, 67, 71-76, 122-123 |
| autogpt/commands/execute\_code.py           |       71 |       71 |       22 |        0 |      0% |     2-186 |
| autogpt/commands/file\_operations.py        |      139 |       50 |       44 |        6 |     63% |48->52, 72->exit, 118-137, 152, 157, 162-163, 188-189, 203, 209-210, 233, 253-284 |
| autogpt/commands/git\_operations.py         |       15 |       15 |        0 |        0 |      0% |      2-35 |
| autogpt/commands/google\_search.py          |       40 |       40 |       14 |        0 |      0% |     2-117 |
| autogpt/commands/image\_gen.py              |       54 |       39 |       16 |        0 |     21% |28-39, 52-77, 91-115, 136-165 |
| autogpt/commands/improve\_code.py           |       10 |       10 |        0 |        0 |      0% |      1-35 |
| autogpt/commands/times.py                   |        3 |        3 |        0 |        0 |      0% |      1-10 |
| autogpt/commands/twitter.py                 |       19 |       19 |        0 |        0 |      0% |      2-44 |
| autogpt/commands/web\_playwright.py         |       41 |       41 |       14 |        0 |      0% |      2-80 |
| autogpt/commands/web\_requests.py           |       66 |       12 |       28 |        6 |     81% |31-35, 106, 110, 123, 144, 149, 172, 185 |
| autogpt/commands/web\_selenium.py           |       70 |       70 |       20 |        0 |      0% |     2-160 |
| autogpt/commands/write\_tests.py            |       10 |       10 |        0 |        0 |      0% |      2-37 |
| autogpt/config/\_\_init\_\_.py              |        4 |        0 |        0 |        0 |    100% |           |
| autogpt/config/ai\_config.py                |       59 |       43 |       16 |        0 |     21% |44-50, 67-77, 91-97, 113-152 |
| autogpt/config/config.py                    |      142 |       34 |       14 |        2 |     71% |52-55, 134, 149-162, 177-184, 192, 216, 220, 224, 228, 232, 236, 240, 244, 248, 256, 261-268 |
| autogpt/config/singleton.py                 |        9 |        0 |        2 |        0 |    100% |           |
| autogpt/configurator.py                     |       61 |       61 |       30 |        0 |      0% |     2-134 |
| autogpt/json\_utils/\_\_init\_\_.py         |        0 |        0 |        0 |        0 |    100% |           |
| autogpt/json\_utils/json\_fix\_general.py   |       66 |       28 |       22 |        6 |     55% |27-39, 57-58, 61-62, 66, 81, 88, 101->103, 104, 106->108, 110, 115-121, 123 |
| autogpt/json\_utils/json\_fix\_llm.py       |       85 |       51 |       28 |        2 |     34% |51-82, 96-112, 133->exit, 171-186, 190-220 |
| autogpt/json\_utils/utilities.py            |       25 |       17 |       13 |        0 |     21% |24-28, 37-54 |
| autogpt/llm\_utils.py                       |       80 |       66 |       48 |        0 |     11% |36-51, 73-153, 158-185 |
| autogpt/logs.py                             |      190 |      113 |       56 |        5 |     33% |30->33, 84-93, 103, 114, 117->120, 119, 123-124, 127-135, 145-162, 170-171, 189, 193, 207-294, 300-332 |
| autogpt/memory/\_\_init\_\_.py              |       53 |       23 |       24 |        7 |     48% |12-14, 20-22, 27, 35, 44-52, 54-60, 62-68, 70-76, 78, 80->84, 83, 88 |
| autogpt/memory/base.py                      |       25 |        9 |        2 |        0 |     59% |12-19, 27, 31, 35, 39, 43 |
| autogpt/memory/local.py                     |       57 |       30 |       14 |        1 |     42% |43-54, 72-91, 99-100, 111, 124-130, 136 |
| autogpt/memory/milvus.py                    |       33 |       32 |        4 |        0 |      3% |     4-115 |
| autogpt/memory/no\_memory.py                |       16 |        6 |        0 |        0 |     62% |23, 34, 46, 54, 67, 73 |
| autogpt/memory/pinecone.py                  |       42 |       30 |        6 |        0 |     25% |11-44, 47-52, 55, 58-59, 67-72, 75 |
| autogpt/memory/redismem.py                  |       65 |       46 |        6 |        0 |     27% |37-78, 89-102, 113, 121-122, 133-150, 156 |
| autogpt/memory/weaviate.py                  |       61 |       59 |       16 |        0 |      3% |     4-127 |
| autogpt/models/base\_open\_ai\_plugin.py    |       54 |        0 |        0 |        0 |    100% |           |
| autogpt/permanent\_memory/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| autogpt/permanent\_memory/sqlite3\_store.py |       85 |       85 |       20 |        0 |      0% |     1-118 |
| autogpt/plugins.py                          |      132 |       23 |       65 |       15 |     79% |33->32, 38-40, 73-76, 78-81, 86-91, 100-101, 119-121, 140->174, 153->166, 161-165, 190-193, 210->209, 213->215, 231->240, 236->235, 240->242 |
| autogpt/processing/\_\_init\_\_.py          |        0 |        0 |        0 |        0 |    100% |           |
| autogpt/processing/html.py                  |        7 |        0 |        4 |        0 |    100% |           |
| autogpt/processing/text.py                  |       64 |       50 |       18 |        0 |     17% |34-69, 73, 90-138, 154-156, 169 |
| autogpt/prompts/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| autogpt/prompts/generator.py                |       39 |        2 |       18 |        2 |     89% |   64, 125 |
| autogpt/prompts/prompt.py                   |       42 |       42 |       10 |        0 |      0% |     1-110 |
| autogpt/setup.py                            |       29 |       29 |       10 |        0 |      0% |      2-77 |
| autogpt/speech/\_\_init\_\_.py              |        2 |        0 |        0 |        0 |    100% |           |
| autogpt/speech/base.py                      |       20 |        4 |        2 |        0 |     73% |32-33, 40, 50 |
| autogpt/speech/brian.py                     |       19 |       12 |        4 |        0 |     30% | 15, 26-40 |
| autogpt/speech/eleven\_labs.py              |       34 |       24 |       10 |        0 |     23% |23-46, 59-60, 72-86 |
| autogpt/speech/gtts.py                      |       13 |        5 |        0 |        0 |     62% |     18-22 |
| autogpt/speech/macos\_tts.py                |       12 |        7 |        4 |        0 |     31% | 11, 15-21 |
| autogpt/speech/say.py                       |       27 |       11 |        8 |        3 |     54% |15, 17, 19, 32-41 |
| autogpt/spinner.py                          |       33 |       23 |        4 |        0 |     27% |18-22, 26-30, 34-38, 48-52, 60-65 |
| autogpt/token\_counter.py                   |       34 |        2 |       14 |        2 |     92% |    35, 38 |
| autogpt/types/openai.py                     |        4 |        0 |        0 |        0 |    100% |           |
| autogpt/utils.py                            |       56 |       43 |       14 |        0 |     19% |11-12, 16-21, 25-36, 45-49, 53-60, 64-69, 73-83 |
| autogpt/workspace.py                        |       16 |        0 |        4 |        1 |     95% |    14->18 |
|                                   **TOTAL** | **2849** | **1776** |  **846** |   **61** | **34%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Significant-Gravitas/Auto-GPT/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Significant-Gravitas/Auto-GPT/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Significant-Gravitas/Auto-GPT/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Significant-Gravitas/Auto-GPT/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FSignificant-Gravitas%2FAuto-GPT%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Significant-Gravitas/Auto-GPT/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.