# Google Maps Search

## What it is
A block that searches for local businesses using the Google Maps API.

## What it does
This block allows users to search for places of interest, such as restaurants, shops, or attractions, within a specified area using Google Maps data.

## How it works
The block takes a search query, location details, and API credentials as input. It then communicates with the Google Maps API to fetch information about relevant places. The results are processed and returned as structured data containing details about each place found.

## Inputs
| Input | Description |
|-------|-------------|
| API Key | A secret key required to authenticate and use the Google Maps API |
| Query | The search term for finding local businesses (e.g., "restaurants in New York") |
| Radius | The search area radius in meters, with a maximum of 50,000 meters (about 31 miles) |
| Max Results | The maximum number of places to return, up to 60 results |

## Outputs
| Output | Description |
|--------|-------------|
| Place | Information about a found place, including: |
| - Name | The name of the business or location |
| - Address | The full address of the place |
| - Phone | The contact phone number |
| - Rating | The average rating (out of 5) given by users |
| - Reviews | The total number of user reviews |
| - Website | The official website of the place, if available |
| Error | A message describing any issues that occurred during the search process |

## Possible use case
A travel planning application could use this block to help users discover popular restaurants, attractions, or accommodations in their destination city. By inputting a search query like "family-friendly restaurants in Paris" and specifying a search radius around their hotel, travelers could quickly get a list of suitable dining options with ratings, contact information, and websites for making reservations.