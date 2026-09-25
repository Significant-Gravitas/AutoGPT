# Search
<!-- MANUAL: file_description -->
Blocks for web searching, content extraction, and information retrieval from various search engines and APIs.
<!-- END MANUAL -->

## Get Wikipedia Summary

### What it is
This block fetches the summary of a given topic from Wikipedia.

### How it works
<!-- MANUAL: how_it_works -->
The block sends a request to Wikipedia's API with the provided topic. It then extracts the summary from the response and returns it. If there's an error during this process, it will return an error message instead.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| topic | The topic to fetch the summary for | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the summary cannot be retrieved | str |
| summary | The summary of the given topic | str |

### Possible use case
<!-- MANUAL: use_case -->
A student researching for a project could use this block to quickly get overviews of various topics, helping them decide which areas to focus on for more in-depth study.
<!-- END MANUAL -->

---

## Google Maps Get Directions

### What it is
Get directions between two places with Google Maps: distance, travel time and a route summary, plus turn-by-turn steps if you ask for them. Works for driving, walking, cycling, public transport and two-wheelers, with optional live traffic.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Routes API `computeRoutes` method with an `X-Goog-FieldMask` header, so Google sends back only the fields the block outputs, and the turn-by-turn steps only when `include_steps` is on. The origin and destination go to Google as given: an address or place name, `latitude,longitude` coordinates, or a place ID (`ChIJ...` or `place_id:...`). The first route Google returns supplies the distance, the travel time and a summary such as `14 mins (4.6 km) via Voie Georges Pompidou`. Transit steps also carry the line, stops and times.

Live traffic sets Google's `TRAFFIC_AWARE` routing preference, which only works for drive and two-wheeler routes, so the block refuses it for other modes. A departure time is used for transit timetables and live-traffic predictions. If Google finds no route, the block fails and says so. The Maps API key's Google Cloud project needs the Routes API enabled. Google bills live-traffic and two-wheeler routes at its higher Pro and Enterprise rates.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| origin | Where the route starts: an address, a place name, 'latitude,longitude' or a Google Maps place ID | str | Yes |
| destination | Where the route ends, in the same forms as the origin | str | Yes |
| travel_mode | drive, walk, bicycle, transit (public transport) or two_wheeler (motorbikes and scooters, only in some countries). Google bills two-wheeler routes at a higher rate. | "drive" \| "walk" \| "bicycle" \| "transit" \| "two_wheeler" | No |
| include_steps | Also return turn-by-turn steps | bool | No |
| use_live_traffic | Use live traffic for drive and two-wheeler routes, for more accurate travel times. Google bills these at a higher rate. | bool | No |
| units | metric (km) or imperial (miles), for the distance and time text | "metric" \| "imperial" | No |
| departure_time | When to leave, for transit timetables or live-traffic predictions. Defaults to now. A time without a time zone is read as UTC. | str (date-time) | No |
| language_code | Language for the directions, e.g. 'en' or 'fr' | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| summary | Travel time, distance and main roads, e.g. '14 mins (4.6 km) via Voie Georges Pompidou' | str |
| distance_meters | Length of the route in meters | int |
| distance_text | Length of the route as text, e.g. '4.6 km' or '2.9 mi' | str |
| duration_seconds | Travel time in seconds, including traffic when live traffic is on | int |
| duration_text | Travel time as text, e.g. '14 mins' | str |
| steps | Turn-by-turn steps, when include_steps is on | List[RouteStep] |
| step | Each step, when include_steps is on | RouteStep |
| warnings | Warnings to show with the route, e.g. that walking directions are in beta | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Travel Time Estimates**: Tell a customer how long the drive from their address to your store takes before they book a visit.

**Commute Comparisons**: Compare driving, transit and cycling times from a job candidate's home to the office.

**Walking Directions for Guests**: Send event guests step-by-step walking directions from the nearest station to the venue.
<!-- END MANUAL -->

---

## Google Maps Resolve Links

### What it is
Find the place a Google Maps link points to, including short share links: place ID, name, address, coordinates and types. Works with google.com/maps and maps.app.goo.gl links, up to 20 at once.

### How it works
<!-- MANUAL: how_it_works -->
Short `maps.app.goo.gl` and `goo.gl/maps` links are expanded by following their redirects one hop at a time through AutoGPT's request client, which blocks private and internal addresses. Only short-link hosts are fetched; the Google Maps page itself never is. The block then reads the full URL: a place ID (`query_place_id` or `q=place_id:...`), the place name in `/maps/place/...`, the place's own pin (`!3d...!4d...`) or the map's centre (`@lat,lng`).

With `look_up_place` on, the block asks the Places API (New) for the place's details, either directly by place ID or by searching for the name within about 1 km of the link's pin, and returns its place ID, name, address, coordinates, types and Maps link. Links to a plain map view come back as coordinates only. Directions links, links that only carry Google's internal CID, and expired short links are listed in `failed` with the reason, while the other links still resolve. The Maps API key's Google Cloud project needs the Places API (New) enabled for lookups.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | Google Maps links to resolve, up to 20: google.com/maps links or maps.app.goo.gl short links | List[str] | Yes |
| look_up_place | Look up each linked place for its place ID, address and types. Turn off to only expand and read the links, which is free. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | What each link points at, in the order given | List[ResolvedMapsLink] |
| result | Each resolved link | ResolvedMapsLink |
| failed | Links that couldn't be resolved, with the reason | List[FailedMapsLink] |

### Possible use case
<!-- MANUAL: use_case -->
**Shared Location Intake**: Turn the Google Maps link a customer pastes into a form into an address and coordinates you can store.

**Places From Group Chats**: Resolve the short Maps links collected in a team chat into named places for a trip itinerary.

**Directions From a Shared Pin**: Get the place ID behind a shared link and pass it to Google Maps Get Directions.
<!-- END MANUAL -->

---

## Google Maps Resolve Places

### What it is
Look up place names or addresses on Google Maps and get each one's place ID, name, full address, coordinates, types and Google Maps link. Takes up to 20 at once.

### How it works
<!-- MANUAL: how_it_works -->
For each query, the block runs a Places API (New) Text Search that asks only for the best match's place ID, which Google doesn't charge for, then reads that place's details: name, full address, coordinates, types and Google Maps link. Queries that are already place IDs skip the search. The queries run in parallel, and the places come back in query order, each with the query it was found for.

Queries that match nothing are listed in `unresolved`, so one bad query doesn't fail the rest. An API problem, such as the Places API (New) not being enabled on the key's Google Cloud project, fails the block with a message that says what to fix. `region_code` prefers matches in one country.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| queries | Place names or addresses to look up, up to 20. Be specific, e.g. 'Eiffel Tower, Paris' or '1600 Amphitheatre Pkwy, Mountain View, CA'. Searches like 'coffee near me' or chain names like 'Starbucks' don't name one place. | List[str] | Yes |
| region_code | Two-letter country code to prefer matches in, e.g. 'US' or 'GB' | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| places | The places found, in the order of the queries | List[ResolvedPlace] |
| place | Each place found | ResolvedPlace |
| unresolved | Queries that matched no place | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Address Book Cleanup**: Turn a column of business names from a spreadsheet into full addresses and coordinates.

**Place IDs for Other Blocks**: Resolve landmark names to place IDs before getting the weather or directions for them.

**Itinerary Links**: Match each stop in a travel plan to its Google Maps link to share with travellers.
<!-- END MANUAL -->

---

## Google Maps Search

### What it is
This block searches for local businesses using Google Maps API.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Google Maps Places API to search for businesses and locations based on a query. Configure radius (up to 50km) to limit the search area and max_results (up to 60) to control how many places are returned.

Each place result includes name, address, rating, reviews, and geographic coordinates for integration with mapping or navigation workflows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query for local businesses | str | Yes |
| radius | Search radius in meters (max 50000) | int | No |
| max_results | Maximum number of results to return (max 60) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| place | Place found | Place |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Generation**: Find businesses in a specific area for sales outreach.

**Competitive Analysis**: Search for competitors in target locations to analyze their presence and ratings.

**Local SEO**: Gather data on local businesses for market research or directory building.
<!-- END MANUAL -->

---

## Google Maps Weather

### What it is
Get the weather for a place from Google Maps: current conditions, a daily forecast for up to 10 days, or an hourly forecast for up to 240 hours. The place can be an address, a place name, coordinates or a place ID.

### How it works
<!-- MANUAL: how_it_works -->
The block first turns the location into coordinates. `latitude,longitude` text is used as given; anything else is looked up with the Places API (New), using a free ID-only search followed by a Place Details request for the address and coordinates. It then calls the Weather API: `currentConditions:lookup` in current mode, `forecast/days:lookup` in daily mode (up to 10 days in one request), or `forecast/hours:lookup` in hourly mode, following its 24-hour pages for up to 240 hours.

Only the outputs for the chosen mode are filled: `current`, `days` and `day`, or `hours` and `hour`. Every mode also returns `location` and `time_zone`. The numbers follow the `units` input. The Weather API has no data for some countries, including China, Japan and South Korea; Google's error is passed on. The Maps API key's Google Cloud project needs the Weather API enabled, and the Places API (New) too unless you give coordinates.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| location | Where to get the weather for: an address, a place name, 'latitude,longitude' or a Google Maps place ID | str | Yes |
| mode | current: the weather right now. daily: a forecast for each day. hourly: a forecast for each hour. | "current" \| "daily" \| "hourly" | No |
| forecast_days | Days to forecast, starting today (daily mode) | int | No |
| forecast_hours | Hours to forecast, starting with the current hour (hourly mode) | int | No |
| units | metric (°C, km/h, mm, km) or imperial (°F, mph, inches, miles) | "metric" \| "imperial" | No |
| language_code | Language for weather descriptions and the address, e.g. 'en' or 'fr'. Defaults to English. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| current | The weather right now (current mode) | CurrentWeather |
| days | One forecast per day, starting today (daily mode) | List[DailyForecast] |
| day | Each day's forecast (daily mode) | DailyForecast |
| hours | One forecast per hour, starting with the current hour (hourly mode) | List[HourlyForecast] |
| hour | Each hour's forecast (hourly mode) | HourlyForecast |
| location | The place the weather is for, with its coordinates | MapsLocation |
| time_zone | The location's time zone, e.g. Europe/Paris | str |

### Possible use case
<!-- MANUAL: use_case -->
**Event Day Planning**: Check the daily forecast at an outdoor event's venue and warn attendees if rain is likely.

**Field Team Scheduling**: Look at the next 12 hours of weather at each job site before assigning outdoor work.

**Travel Briefings**: Add the current conditions at a traveller's destination to their morning itinerary email.
<!-- END MANUAL -->

---
