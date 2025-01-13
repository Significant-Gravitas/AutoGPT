
<file_name>autogpt_platform/backend/backend/blocks/search.md</file_name>

## Get Wikipedia Summary

### What it is
A specialized search block that retrieves summary information from Wikipedia articles.

### What it does
This block fetches and provides concise summaries of Wikipedia articles based on a given topic, making it easy to get quick information about any subject available on Wikipedia.

### How it works
The block takes a topic from the user, connects to Wikipedia's API, retrieves the article summary for that topic, and returns the extracted information. If the article cannot be found or there's an error, it provides an appropriate error message.

### Inputs
- Topic: The subject or article title you want to get information about from Wikipedia

### Outputs
- Summary: The extracted summary text from the Wikipedia article about the requested topic
- Error: A message explaining what went wrong if the summary couldn't be retrieved

### Possible use case
A student researching different topics for a school project could use this block to quickly get overviews of various subjects without having to read entire Wikipedia articles.

## Get Weather Information

### What it is
A weather information retrieval block that connects to the OpenWeatherMap service to provide current weather data for any location.

### What it does
This block fetches real-time weather information including temperature, humidity, and current weather conditions for a specified location using the OpenWeatherMap API.

### How it works
The block takes a location name and connects to OpenWeatherMap's API using provided credentials. It retrieves current weather data and converts it into an easy-to-read format, with the option to display temperature in either Celsius or Fahrenheit.

### Inputs
- Location: The name of the place you want to get weather information for (e.g., "New York", "London", "Tokyo")
- Credentials: OpenWeatherMap API authentication details (automatically handled by the system)
- Use Celsius: Option to choose between Celsius (true) or Fahrenheit (false) for temperature display (defaults to Celsius)

### Outputs
- Temperature: The current temperature at the specified location
- Humidity: The current humidity percentage at the specified location
- Condition: A description of the current weather conditions (e.g., "overcast clouds", "clear sky")
- Error: A message explaining what went wrong if the weather information couldn't be retrieved

### Possible use case
A travel planning application could use this block to provide users with current weather information for their destination cities, helping them pack appropriate clothing and plan outdoor activities.

