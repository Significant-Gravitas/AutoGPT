import requests
from autogpt_server.data.block import Block, BlockSchema, BlockOutput

class getopenweathermapweather(Block):
    class Input(BlockSchema):
        location: str
        api_key: str
        use_celsius: bool

    class Output(BlockSchema):
        temperature: str
        humidity: str
        condition: str

    def __init__(self):
        super().__init__(
            id="f7a8b2c3-6d4e-5f8b-9e7f-6d4e5f8b9e7f",
            input_schema=getopenweathermapweather.Input,
            output_schema=getopenweathermapweather.Output,
            test_input={"location": "New York", "api_key": "YOUR_API_KEY", "use_celsius": True},
            test_output={"temperature": "23.5", "humidity": "60", "condition": "Sunny"},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            units = "metric" if input_data.use_celsius else "imperial"
            response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={input_data.location}&appid={input_data.api_key}&units={units}")
            response.raise_for_status()
            weather_data = response.json()
            
            # Check if required data is in the response
            if 'main' in weather_data and 'weather' in weather_data:
                yield "temperature", str(weather_data['main']['temp'])
                yield "humidity", str(weather_data['main']['humidity'])
                yield "condition", weather_data['weather'][0]['description']
            else:
                raise KeyError(f"Expected keys not found in response: {weather_data}")

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 403:
                raise ValueError(f"Request to weather API failed: 403 Forbidden. Check your API key and permissions.")
            else:
                raise ValueError(f"HTTP error occurred: {http_err}")
        except requests.RequestException as e:
            raise ValueError(f"Request to weather API failed: {e}")
        except KeyError as e:
            raise ValueError(f"Error processing weather data: {e}")
