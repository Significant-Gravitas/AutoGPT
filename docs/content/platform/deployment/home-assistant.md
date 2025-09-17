# AutoGPT Platform Home Assistant Add-on

This guide covers integrating AutoGPT Platform with Home Assistant as an add-on.

## Overview

The AutoGPT Platform Home Assistant Add-on allows you to run AutoGPT Platform directly within your Home Assistant environment, enabling powerful automation capabilities and seamless integration with your smart home.

## Prerequisites

- **Home Assistant OS** or **Home Assistant Supervised**
- **Advanced Mode** enabled in Home Assistant
- **Minimum 4GB RAM** available
- **10GB+ free storage space**

## Installation

### Method 1: Add-on Store (Coming Soon)

1. Open Home Assistant
2. Go to **Settings** → **Add-ons**
3. Click **Add-on Store**
4. Search for "AutoGPT Platform"
5. Click **Install**

### Method 2: Manual Repository Addition

Until the add-on is available in the official store:

1. Go to **Settings** → **Add-ons**
2. Click **Add-on Store**
3. Click the three dots (⋮) → **Repositories**
4. Add repository: `https://github.com/Significant-Gravitas/AutoGPT-HomeAssistant-Addon`
5. Find "AutoGPT Platform" in the store
6. Click **Install**

## Configuration

### Basic Configuration

After installation, configure the add-on:

```yaml
# Add-on Configuration
database:
  host: "localhost"
  port: 5432
  username: "autogpt"
  password: "!secret autogpt_db_password"
  database: "autogpt"

redis:
  host: "localhost"
  port: 6379
  password: "!secret autogpt_redis_password"

auth:
  jwt_secret: "!secret autogpt_jwt_secret"
  admin_email: "admin@yourdomain.com"

network:
  backend_port: 8000
  frontend_port: 3000

# Home Assistant Integration
homeassistant:
  enabled: true
  token: "!secret ha_long_lived_token"
  api_url: "http://supervisor/core/api"
```

### Secrets Configuration

Add to your `secrets.yaml`:

```yaml
autogpt_db_password: "your_secure_database_password"
autogpt_redis_password: "your_secure_redis_password"
autogpt_jwt_secret: "your_long_random_jwt_secret"
ha_long_lived_token: "your_home_assistant_long_lived_access_token"
```

## Home Assistant Integration Features

### Available Services

The add-on exposes several services to Home Assistant:

#### autogpt_platform.create_agent
Create a new agent workflow:
```yaml
service: autogpt_platform.create_agent
data:
  name: "Temperature Monitor"
  description: "Monitor temperature and adjust HVAC"
  triggers:
    - type: "state_change"
      entity_id: "sensor.living_room_temperature"
  actions:
    - type: "service_call"
      service: "climate.set_temperature"
```

#### autogpt_platform.run_agent
Execute an agent workflow:
```yaml
service: autogpt_platform.run_agent
data:
  agent_id: "temperature_monitor_001"
  input_data:
    current_temp: "{{ states('sensor.living_room_temperature') }}"
```

#### autogpt_platform.stop_agent
Stop a running agent:
```yaml
service: autogpt_platform.stop_agent
data:
  agent_id: "temperature_monitor_001"
```

### Entity Integration

The add-on creates several entities in Home Assistant:

#### Sensors
- `sensor.autogpt_agent_count` - Number of active agents
- `sensor.autogpt_task_queue` - Tasks in queue
- `sensor.autogpt_system_status` - Overall system health

#### Binary Sensors
- `binary_sensor.autogpt_backend_online` - Backend service status
- `binary_sensor.autogpt_database_connected` - Database connection status

#### Switches
- `switch.autogpt_agent_execution` - Enable/disable agent execution
- `switch.autogpt_auto_updates` - Enable/disable automatic updates

### Automation Examples

#### Example 1: Smart Lighting Based on Occupancy
```yaml
automation:
  - alias: "AutoGPT Smart Lighting"
    trigger:
      - platform: state
        entity_id: binary_sensor.living_room_occupancy
    action:
      - service: autogpt_platform.run_agent
        data:
          agent_id: "smart_lighting_001"
          input_data:
            occupancy: "{{ trigger.to_state.state }}"
            time_of_day: "{{ now().hour }}"
            current_brightness: "{{ state_attr('light.living_room', 'brightness') }}"
```

#### Example 2: Energy Management
```yaml
automation:
  - alias: "AutoGPT Energy Optimization"
    trigger:
      - platform: time_pattern
        minutes: "/15"  # Every 15 minutes
    action:
      - service: autogpt_platform.run_agent
        data:
          agent_id: "energy_optimizer_001"
          input_data:
            current_usage: "{{ states('sensor.home_energy_usage') }}"
            solar_production: "{{ states('sensor.solar_power') }}"
            electricity_price: "{{ states('sensor.electricity_price') }}"
```

## Advanced Configuration

### Custom Blocks for Home Assistant

You can create custom blocks that interact with Home Assistant:

```python
# Example: Home Assistant Service Call Block
class HomeAssistantServiceBlock(Block):
    def __init__(self):
        super().__init__(
            id="ha_service_call",
            description="Call a Home Assistant service",
            input_schema=self.Input,
            output_schema=self.Output,
        )
    
    class Input(BlockSchema):
        service: str = Field(description="Service to call (e.g., 'light.turn_on')")
        entity_id: str = Field(description="Target entity ID")
        service_data: dict = Field(description="Service data", default={})
    
    class Output(BlockSchema):
        success: bool = Field(description="Service call success")
        response: dict = Field(description="Service response data")
    
    async def run(self, input_data: Input) -> BlockOutput:
        # Implementation calls Home Assistant API
        pass
```

### Resource Limits

Configure resource limits in the add-on:

```yaml
resources:
  cpu: "2"
  memory: "4Gi"
  
limits:
  max_agents: 50
  max_executions_per_hour: 1000
```

## Networking

### Internal Access
- **Backend API**: `http://127.0.0.1:8000`
- **Frontend**: `http://127.0.0.1:3000`
- **WebSocket**: `ws://127.0.0.1:8001`

### External Access
If you need external access, configure the add-on to use host networking:

```yaml
network:
  external_access: true
  host_networking: false  # Use bridge mode
```

Then access via:
- **Frontend**: `http://[HA_IP]:3000`
- **Backend API**: `http://[HA_IP]:8000`

## Backup and Restore

### Automatic Backups

The add-on integrates with Home Assistant's backup system:

1. Go to **Settings** → **System** → **Backups**
2. Click **Create Backup**
3. Select **AutoGPT Platform** in partial backup options

### Manual Backup

Create manual backups of important data:

```bash
# From Home Assistant Terminal add-on
ha addons backup autogpt-platform
```

### Restore Process

1. **Stop the add-on**: Settings → Add-ons → AutoGPT Platform → Stop
2. **Restore backup**: System → Backups → Select backup → Restore
3. **Start the add-on**: Settings → Add-ons → AutoGPT Platform → Start

## Monitoring and Logs

### Log Access

View logs through Home Assistant:

1. Go to **Settings** → **Add-ons**
2. Select **AutoGPT Platform**
3. Click **Logs** tab

### Log Levels

Configure logging in the add-on:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file_logging: true
  max_log_size: "100MB"
```

### Performance Monitoring

Monitor add-on performance:

```yaml
# Lovelace dashboard card
type: entities
title: AutoGPT Platform Status
entities:
  - sensor.autogpt_agent_count
  - sensor.autogpt_task_queue
  - sensor.autogpt_system_status
  - binary_sensor.autogpt_backend_online
  - switch.autogpt_agent_execution
```

## Troubleshooting

### Common Issues

1. **Add-on won't start**
   - Check available memory (minimum 4GB required)
   - Verify configuration syntax
   - Check logs for specific errors

2. **Home Assistant integration not working**
   - Verify long-lived access token
   - Check Home Assistant API permissions
   - Ensure WebSocket connection is established

3. **Agents not executing**
   - Check `switch.autogpt_agent_execution` is enabled
   - Verify database connectivity
   - Check agent configuration and triggers

### Debug Mode

Enable debug mode for detailed logging:

```yaml
debug: true
logging:
  level: "DEBUG"
```

### Performance Issues

If experiencing performance issues:

1. **Increase memory allocation**:
   ```yaml
   resources:
     memory: "6Gi"  # Increase from default 4Gi
   ```

2. **Limit concurrent executions**:
   ```yaml
   limits:
     max_concurrent_agents: 10
   ```

3. **Optimize database queries**:
   ```yaml
   database:
     connection_pool_size: 20
     max_connections: 100
   ```

## Security Considerations

1. **Use strong secrets** - Generate random passwords and JWT secrets
2. **Limit network access** - Use internal networking when possible
3. **Regular updates** - Keep the add-on updated
4. **Backup encryption** - Enable backup encryption in Home Assistant
5. **Review agent permissions** - Audit what services agents can access

## Updates

The add-on supports automatic updates through Home Assistant:

1. **Enable auto-updates**: Add-on settings → Auto-update
2. **Manual updates**: Add-on store → AutoGPT Platform → Update

## Support and Community

- **Home Assistant Community**: [AutoGPT Platform Discussion](https://community.home-assistant.io)
- **GitHub Issues**: [AutoGPT Repository](https://github.com/Significant-Gravitas/AutoGPT/issues)
- **Add-on Repository**: [AutoGPT HomeAssistant Add-on](https://github.com/Significant-Gravitas/AutoGPT-HomeAssistant-Addon)

## Examples and Templates

Check the [examples directory](https://github.com/Significant-Gravitas/AutoGPT-HomeAssistant-Addon/tree/main/examples) for:

- Sample automation configurations
- Custom block examples
- Integration templates
- Best practices guides