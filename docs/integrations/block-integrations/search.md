# Search

## Get Wikipedia Summary

### What It Is

A block that retrieves a summary of a given topic from Wikipedia.

### What It Does

This block takes a topic as input and fetches a concise summary about that topic from Wikipedia's API.

### How It Works

The block sends a request to Wikipedia's API with the provided topic. It then extracts the summary from the response and returns it. If there's an error during this process, it will return an error message instead.

### Inputs

| Input | Description                                                |
| ----- | ---------------------------------------------------------- |
| Topic | The subject you want to get a summary about from Wikipedia |

### Outputs

| Output  | Description                                            |
| ------- | ------------------------------------------------------ |
| Summary | A brief overview of the requested topic from Wikipedia |
| Error   | An error message if the summary retrieval fails        |

### Possible Use Case

A student researching for a project could use this block to quickly get overviews of various topics, helping them decide which areas to focus on for more in-depth study.

***

## Search The Web

### What It Is

A block that performs web searches and returns the results.

### What it Does

This block takes a search query and returns a list of relevant web pages, including their titles, URLs, and brief descriptions.

### How It Works

The block sends the search query to a search engine API, processes the results, and returns them in a structured format.

### Inputs

| Input             | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| Query             | The search term or phrase to look up on the web                |
| Number of Results | How many search results to return (optional, default may vary) |

### Outputs

| Output  | Description                                                             |
| ------- | ----------------------------------------------------------------------- |
| Results | A list of search results, each containing a title, URL, and description |
| Error   | An error message if the search fails                                    |

### Possible Use Case

A content creator could use this block to research trending topics in their field, gathering ideas for new articles or videos.

***

## Extract Website Content

### What It Is

A block that retrieves and extracts content from specified websites.

### What it Does

This block takes a URL as input, visits the webpage, and extracts the main content, removing navigation elements, ads, and other non-essential parts.

### How It Works

The block sends a request to the given URL, downloads the HTML content, and uses content extraction algorithms to identify and extract the main text content of the page.

### Inputs

| Input | Description                                         |
| ----- | --------------------------------------------------- |
| URL   | The web address of the page to extract content from |

### Outputs

| Output  | Description                                      |
| ------- | ------------------------------------------------ |
| Content | The main text content extracted from the webpage |
| Title   | The title of the webpage                         |
| Error   | An error message if the content extraction fails |

### Possible Use Case

A data analyst could use this block to automatically extract article content from news websites for sentiment analysis or topic modeling.

***

## Get Weather Information

### What It Is

A block that fetches current weather data for a specified location.

### What it Does

This block takes a location name as input and returns current weather information such as temperature, humidity, and weather conditions.

### How It Works

The block sends a request to a weather API (like OpenWeatherMap) with the provided location. It then processes the response to extract relevant weather data.

### Inputs

| Input       | Description                                                                      |
| ----------- | -------------------------------------------------------------------------------- |
| Location    | The city or area you want to get weather information for                         |
| API Key     | Your personal OpenWeatherMap API key (this is kept secret)                       |
| Use Celsius | An option to choose between Celsius (true) or Fahrenheit (false) for temperature |

### Outputs

| Output      | Description                                                              |
| ----------- | ------------------------------------------------------------------------ |
| Temperature | The current temperature in the specified location                        |
| Humidity    | The current humidity percentage in the specified location                |
| Condition   | A description of the current weather condition (e.g., "overcast clouds") |
| Error       | A message explaining what went wrong if the weather data retrieval fails |

### Possible Use Case

A travel planning application could use this block to provide users with current weather information for their destination cities.
