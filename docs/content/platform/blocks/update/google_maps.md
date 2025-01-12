
## Google Maps Search

### What it is
A specialized search tool that connects with Google Maps to find and retrieve information about local businesses and places.

### What it does
This block searches for places and businesses using the Google Maps platform, providing detailed information about each location including its name, address, phone number, rating, number of reviews, and website.

### How it works
The block takes a search query and parameters from the user, connects to Google Maps using an API key, and searches for matching locations within the specified radius. For each found location, it gathers detailed information and returns it in an organized format. The search continues until it reaches either the maximum number of requested results or exhausts all available matches.

### Inputs
- Google Maps API Key: Authentication credentials required to access the Google Maps service
- Search Query: Text describing what you're looking for (e.g., "restaurants in New York")
- Search Radius: The distance (in meters) from a central point to search within, up to 50,000 meters
- Maximum Results: The maximum number of places you want to receive, up to 60 results

### Outputs
- Place: Detailed information about each found location, including:
  - Name: The business or location name
  - Address: Complete formatted address
  - Phone: Contact phone number
  - Rating: Average rating from Google Maps reviews
  - Reviews: Total number of reviews
  - Website: URL of the business website
- Error: A message explaining what went wrong if the search fails

### Possible use case
A restaurant discovery application that helps users find dining options in their area. The application could use this block to search for "Italian restaurants" within a 5-kilometer radius of the user's location, returning details about each restaurant including ratings, address, and contact information. This would help users make informed decisions about where to eat based on location, popularity, and reviews.

