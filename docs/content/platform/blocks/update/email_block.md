
<file_name>autogpt_platform/backend/backend/blocks/google_maps.md</file_name>

## Google Maps Search

### What it is
A specialized search tool that connects with Google Maps to find and retrieve information about local businesses and places.

### What it does
This block searches for local businesses and places using the Google Maps API, providing detailed information about each location including its name, address, phone number, rating, number of reviews, and website.

### How it works
The block takes a search query and parameters from the user, connects to Google Maps using an API key, performs the search within the specified radius, and returns detailed information about each matching place. It can process multiple pages of results until it reaches the requested number of results or exhausts all available matches.

### Inputs
- Credentials: Google Maps API Key required to access the Google Maps service
- Query: The search text describing what you're looking for (e.g., "restaurants in New York")
- Radius: How far to search from the target location, in meters (between 1 and 50,000 meters)
- Max Results: Maximum number of places to return (between 1 and 60 results)

### Outputs
- Place: For each location found, returns a structured set of information including:
  - Name: The business or place name
  - Address: Complete formatted address
  - Phone: Contact phone number
  - Rating: Average rating (out of 5)
  - Reviews: Total number of user reviews
  - Website: URL of the business website
- Error: If the search encounters any problems, an error message is provided

### Possible use case
A travel planning application that needs to find highly-rated restaurants within walking distance of a hotel. The block could search for "restaurants" within a 1000-meter radius, returning details about the top 20 establishments, including their ratings, addresses, and contact information. Users could then easily compare options and make reservations.

