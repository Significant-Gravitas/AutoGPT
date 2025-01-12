

# Search Blocks Documentation

## Get Wikipedia Summary

### What it is
A block that retrieves concise summaries of topics from Wikipedia.

### What it does
This block takes a topic as input and fetches a summary description of that topic from Wikipedia's public API, making it easy to get quick, reliable information about various subjects.

### How it works
The block connects to Wikipedia's REST API, searches for the requested topic, and returns a condensed version of the article's introduction. If the topic is found, it provides a summary; if not, it returns an error message.

### Inputs
- Topic: The subject you want to learn about (e.g., "Artificial Intelligence", "Solar System", "Leonardo da Vinci")

### Outputs
- Summary: A concise overview of the requested topic from Wikipedia
- Error: A message explaining what went wrong if the summary couldn't be retrieved

### Possible use case
A student creating a research project could use this block to quickly gather initial information about different subjects, or a content creator could use it to get accurate starting points for article writing.

## Get Weather Information

### What it is
A block that provides current weather information for any location using the OpenWeatherMap service.

### What it does
This block retrieves real-time weather data including temperature, humidity, and current weather conditions for a specified location.

### How it works
The block connects to the OpenWeatherMap API using provided credentials, sends a request with the specified location, and returns current weather details. It can display temperatures in either Celsius or Fahrenheit based on user preference.

### Inputs
- Location: The place you want to get weather information for (e.g., "New York", "London", "Tokyo")
- Use Celsius: A toggle to switch between Celsius (true) and Fahrenheit (false) temperature units
- Credentials: OpenWeatherMap API credentials required to access the weather service

### Outputs
- Temperature: The current temperature in the requested location (in either Celsius or Fahrenheit)
- Humidity: The current humidity percentage in the location
- Condition: A description of the current weather (e.g., "overcast clouds", "light rain")
- Error: A message explaining what went wrong if the weather information couldn't be retrieved

### Possible use case
A travel planning application could use this block to display current weather conditions for different destinations, or a smart home system could use it to adjust indoor settings based on outdoor weather conditions.

