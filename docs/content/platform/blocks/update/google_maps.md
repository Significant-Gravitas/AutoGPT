

## Google Maps Search

### What it is
A specialized search tool that connects with Google Maps to find and retrieve information about local businesses and places.

### What it does
This block searches for places and businesses using the Google Maps API based on user-provided search criteria. It returns detailed information about each found location, including name, address, phone number, rating, review count, and website.

### How it works
The block takes a search query and parameters from the user, connects to Google Maps using an API key, performs the search, and then collects detailed information about each matching location. It processes the results and returns them one at a time, ensuring that users get comprehensive information about each place found.

### Inputs
- Google Maps API Key: Authentication credentials required to access the Google Maps service
- Search Query: Text describing what you're looking for (e.g., "restaurants in New York")
- Search Radius: The distance (in meters) from the center point to search within, up to 50,000 meters (50 km)
- Maximum Results: The maximum number of places to return, up to 60 results

### Outputs
- Place Information: For each location found, returns:
  - Name: The business or location name
  - Address: Complete formatted address
  - Phone: Contact phone number
  - Rating: Average rating (out of 5)
  - Reviews: Total number of user reviews
  - Website: Associated website URL
- Error Message: Information about what went wrong if the search fails

### Possible use cases
- A restaurant discovery app that helps users find dining options in their area
- A real estate application searching for nearby amenities around properties
- A tourism platform helping visitors discover attractions in a specific area
- A business analysis tool gathering information about local competitors
- A local service directory helping users find specific businesses like mechanics or dentists within their preferred radius

