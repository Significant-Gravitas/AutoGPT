
# Commands 

Detailed description of all commands available in Auto-GPT.

### OpenAPI Command

The OpenAPI command is used to generate a OpenAPI client and command from a given Open API definition. 
It's also compatible with OpenAI Plugins as they are effectively are OpenAPI(Swagger) specification,
with OpenAI Plugin metadata on top of it.

To start using it you need to add REST API or OpenAI plugin config to your [openapi_commands.yaml](../openapi_commands.yaml)
and add the name to OPENAPI_APIS in ENVS, so it would be enabled for usage in Auto-GPT.

For any REST API added to the config, following steps are performed:

* OpenAPI(Swagger) spec parsed and each endpoint-method ingested as separate commands
* OpenAI manifest is generated and saved in ./plugins/openapi/{api_name}/ai-plugin.json making it OpenAI plugins generator
* Custom python client package is generated and placed ./plugins/openapi/{api_name}/client
* Making changes in manifest/openapi spec/client anyone can effectively develop new API plugins, or adjust them.

All this making Auto-GPT capable of using any REST API/OpenAI plugin, without any coding. 
