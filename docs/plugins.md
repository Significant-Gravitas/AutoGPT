## Plugins

⚠️💀 **WARNING** 💀⚠️: Review the code of any plugin you use thoroughly, as plugins can execute any Python code, potentially leading to malicious activities, such as stealing your API keys.

To configure plugins, you can create or edit the `plugins_config.yaml` file in the root directory of Auto-GPT. This file allows you to enable or disable plugins as desired. For specific configuration instructions, please refer to the documentation provided for each plugin. The file should be formatted in YAML. Here is an example for your reference:

```yaml
plugin_a:
  config:
    api_key: my-api-key
  enabled: false
plugin_b:
  config: {}
  enabled: true
```

See our [Plugins Repo](https://github.com/Significant-Gravitas/Auto-GPT-Plugins) for more info on how to install all the amazing plugins the community has built!

Alternatively, developers can use the [Auto-GPT Plugin Template](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template) as a starting point for creating your own plugins.

