import os
import json
import pandas as pd
import glob
from gql.transport.aiohttp import AIOHTTPTransport
from gql import gql, Client
import os


def get_reports():
    # Initialize an empty list to store the report data
    report_data = []

    # Get the current working directory
    current_dir = os.getcwd()

    # Check if the current directory ends with 'reports'
    if current_dir.endswith("reports"):
        reports_dir = "/"
    else:
        reports_dir = "reports"

    # Iterate over all agent directories in the reports directory
    for agent_name in os.listdir(reports_dir):
        agent_dir = os.path.join(reports_dir, agent_name)

        # Check if the item is a directory (an agent directory)
        if os.path.isdir(agent_dir):
            # Construct the path to the report.json file
            # Use glob to find all run directories in the agent_dir
            run_dirs = glob.glob(os.path.join(agent_dir, "*"))

            # For each run directory, add the report.json to the end
            report_files = [
                os.path.join(run_dir, "report.json") for run_dir in run_dirs
            ]
            for report_file in report_files:
                # Check if the report.json file exists
                if os.path.isfile(report_file):
                    # Open the report.json file
                    with open(report_file, "r") as f:
                        # Load the JSON data from the file
                        report = json.load(f)

                        # Iterate over all tests in the report
                        for test_name, test_data in report["tests"].items():
                            try:
                                # Append the relevant data to the report_data list
                                if agent_name is not None:
                                    report_data.append(
                                        {
                                            "agent": agent_name.lower(),
                                            "benchmark_start_time": report[
                                                "benchmark_start_time"
                                            ],
                                            "challenge": test_name,
                                            "categories": ", ".join(
                                                test_data["category"]
                                            ),
                                            "task": test_data["task"],
                                            "success": test_data["metrics"]["success"],
                                            "difficulty": test_data["metrics"][
                                                "difficulty"
                                            ],
                                            "success_%": test_data["metrics"][
                                                "success_%"
                                            ],
                                            "run_time": test_data["metrics"][
                                                "run_time"
                                            ],
                                        }
                                    )
                            except KeyError:
                                pass
    return pd.DataFrame(report_data)


def get_helicone_data():
    helicone_api_key = os.getenv("HELICONE_API_KEY")

    url = "https://www.helicone.ai/api/graphql"
    # Replace <KEY> with your personal access key
    transport = AIOHTTPTransport(
        url=url, headers={"authorization": f"Bearer {helicone_api_key}"}
    )

    client = Client(transport=transport, fetch_schema_from_transport=True)

    SIZE = 250

    i = 0

    data = []
    print("Fetching data from Helicone")
    while True:
        query = gql(
            """
            query ExampleQuery($limit: Int, $offset: Int){
                heliconeRequest(
                    limit: $limit
                    offset: $offset
                ) {
                    prompt
                    properties{
                        name
                        value
                    }
                    
                    requestBody
                    response
                    createdAt

                }

                }
        """
        )
        print(f"Fetching {i * SIZE} to {(i + 1) * SIZE} records")
        try:
            result = client.execute(
                query, variable_values={"limit": SIZE, "offset": i * SIZE}
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            result = None

        i += 1

        if result:
            for item in result["heliconeRequest"]:
                properties = {
                    prop["name"]: prop["value"] for prop in item["properties"]
                }
                data.append(
                    {
                        "createdAt": item["createdAt"],
                        "agent": properties.get("agent"),
                        "job_id": properties.get("job_id"),
                        "challenge": properties.get("challenge"),
                        "benchmark_start_time": properties.get("benchmark_start_time"),
                        "prompt": item["prompt"],
                        "model": item["requestBody"].get("model"),
                        "request": item["requestBody"].get("messages"),
                    }
                )

        if not result or (len(result["heliconeRequest"]) == 0):
            print("No more results")
            break

    df = pd.DataFrame(data)
    # Drop rows where agent is None
    df = df.dropna(subset=["agent"])

    # Convert the remaining agent names to lowercase
    df["agent"] = df["agent"].str.lower()

    return df


if os.path.exists("reports_raw.pkl") and os.path.exists("helicone_raw.pkl"):
    reports_df = pd.read_pickle("reports_raw.pkl")
    helicone_df = pd.read_pickle("helicone_raw.pkl")
else:
    reports_df = get_reports()
    reports_df.to_pickle("reports_raw.pkl")
    helicone_df = get_helicone_data()
    helicone_df.to_pickle("helicone_raw.pkl")


def try_formats(date_str):
    formats = ["%Y-%m-%d-%H:%M", "%Y-%m-%dT%H:%M:%S%z"]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    return None


helicone_df["benchmark_start_time"] = pd.to_datetime(
    helicone_df["benchmark_start_time"].apply(try_formats), utc=True
)
helicone_df = helicone_df.dropna(subset=["benchmark_start_time"])
helicone_df["createdAt"] = pd.to_datetime(
    helicone_df["createdAt"], unit="ms", origin="unix"
)
reports_df["benchmark_start_time"] = pd.to_datetime(
    reports_df["benchmark_start_time"].apply(try_formats), utc=True
)
reports_df = reports_df.dropna(subset=["benchmark_start_time"])

assert pd.api.types.is_datetime64_any_dtype(
    helicone_df["benchmark_start_time"]
), "benchmark_start_time in helicone_df is not datetime"
assert pd.api.types.is_datetime64_any_dtype(
    reports_df["benchmark_start_time"]
), "benchmark_start_time in reports_df is not datetime"

reports_df["report_time"] = reports_df["benchmark_start_time"]

df = pd.merge_asof(
    helicone_df.sort_values("benchmark_start_time"),
    reports_df.sort_values("benchmark_start_time"),
    left_on="benchmark_start_time",
    right_on="benchmark_start_time",
    by=["agent", "challenge"],
    direction="backward",
)

df.to_pickle("df.pkl")
print(df.info())
print("Data saved to df.pkl")
print("To load the data use: df = pd.read_pickle('df.pkl')")
