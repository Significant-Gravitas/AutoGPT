
## Google Maps Search

### What it is
A specialized search tool that connects with Google Maps to find and retrieve information about local businesses and places.

### What it does
This block searches for local businesses and places using the Google Maps API, providing detailed information about each location including its name, address, phone number, rating, number of reviews, and website.

### How it works
The block takes a search query and parameters from the user, connects to Google Maps, performs the search within the specified radius, and returns detailed information about each matching place. It can process multiple results and continues searching until it reaches the requested number of results or exhausts all available options.

### Inputs
- Google Maps API Key: Required credentials to access the Google Maps service
- Search Query: The text to search for (e.g., "restaurants in New York")
- Search Radius: The distance (in meters) to search from a central point, with a maximum of 50,000 meters (50 kilometers)
- Maximum Results: The maximum number of places to return, up to 60 results

### Outputs
- Place Information: For each found location, returns:
  - Name: The business or place name
  - Address: The complete formatted address
  - Phone: Contact phone number
  - Rating: Average rating (out of 5)
  - Reviews: Total number of user reviews
  - Website: URL of the business website
- Error: Message explaining what went wrong if the search fails

### Possible use cases
- A restaurant discovery app that helps users find dining options within walking distance
- A real estate application showing nearby amenities around a property
- A tourist guide application helping visitors discover attractions in a new city
- A business analysis tool comparing ratings and reviews of similar businesses in an area
- A local service finder helping users locate specific businesses like mechanics or dentists within their neighborhood

