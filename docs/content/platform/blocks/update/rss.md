
## Get Wikipedia Summary

### What it is
A tool that retrieves concise summaries of topics from Wikipedia.

### What it does
This block fetches and provides a summary of any given topic using Wikipedia's API, making it easy to get quick information about various subjects.

### How it works
When given a topic, the block connects to Wikipedia's service, searches for the topic, and returns a condensed summary of the information. If the topic cannot be found or there's an error, it will provide an error message instead.

### Inputs
- Topic: The subject you want to learn about. This can be any topic that exists on Wikipedia.

### Outputs
- Summary: A concise explanation of the requested topic from Wikipedia
- Error: A message explaining what went wrong if the summary couldn't be retrieved

### Possible use case
A student working on a research project could use this block to quickly get overview information about different subjects they're studying, helping them understand basic concepts before diving deeper into their research.

## Get Weather Information

### What it is
A weather information retrieval tool that uses the OpenWeatherMap service to provide current weather conditions for any location.

### What it does
This block fetches real-time weather data for a specified location, including temperature, humidity, and current weather conditions.

### How it works
When provided with a location and API credentials, the block connects to OpenWeatherMap's service and retrieves current weather information. It can display temperatures in either Celsius or Fahrenheit based on user preference.

### Inputs
- Location: The name of the place you want weather information for (e.g., "New York", "London", "Tokyo")
- Use Celsius: A yes/no option to choose between Celsius (true) or Fahrenheit (false) for temperature display
- Credentials: OpenWeatherMap API authentication details (handled automatically by the system)

### Outputs
- Temperature: The current temperature in the specified location (in either Celsius or Fahrenheit)
- Humidity: The current humidity percentage in the specified location
- Condition: A description of the current weather conditions (e.g., "sunny", "overcast clouds", "light rain")
- Error: A message explaining what went wrong if the weather information couldn't be retrieved

### Possible use case
A travel planning application could use this block to show users the current weather conditions at their destination, helping them pack appropriate clothing and plan suitable activities.
