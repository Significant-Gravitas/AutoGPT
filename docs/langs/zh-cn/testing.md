## 运行测试

要运行所有测试，请运行以下命令：

```
pytest 
```

要仅运行没有集成测试的测试，请运行以下命令：

```
pytest --without-integration
```

要仅运行没有缓慢集成测试的测试，请运行以下命令：

```
pytest --without-slow-integration
```

要运行测试并查看覆盖率，请运行以下命令：

```
pytest --cov=autogpt --without-integration --without-slow-integration
```

## 运行代码检查

此项目使用 [flake8](https://flake8.pycqa.org/en/latest/) 进行代码检查。我们目前使用以下规则：`E303,W293,W291,W292,E305,E231,E302`。有关更多信息，请参见 [flake8 规则](https://www.flake8rules.com/)。

要运行代码检查，请运行以下命令：

```
flake8 autogpt/ tests/

# 或者，如果您想使用与 CI 相同的配置运行 flake8：
flake8 autogpt/ tests/ --select E303,W293,W291,W292,E305,E231,E302
```
