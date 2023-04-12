# Spinner Class

The `Spinner` class is a simple class to display a spinning icon as a loading indicator. It is implemented as a context manager using the `with` statement.

## Basic Usage

To use the `Spinner` class, create an instance of the class and use it as a context manager using the `with` statement. The `message` parameter can be used to specify a custom message to display while the spinner is running.

```python
with Spinner():
    # Your code here
```

## Parameters

- `message` (string): The message to be displayed beside the spinner icon. (default: "Loading...")
- `delay` (float): The delay in seconds between each iteration of the spinner. (default: 0.1)

## Methods

### __init__()

The constructor initializes the spinner with default values. It takes two optional parameters:

- `message` (string): The message to be displayed beside the spinner icon. (default: "Loading...")
- `delay` (float): The delay in seconds between each iteration of the spinner. (default: 0.1)

### spin()

The `spin()` method is called whenever the spinner is run. It displays the spinner icon beside the message and updates it with each iteration of the spinner.

### __enter__()

The `__enter__()` method starts the spinner thread.

### __exit__()

The `__exit__()` method stops the spinner thread and clears the spinner icon and message from the stdout. It also handles any exceptions that occur within the spinner context. 

## Example

```python
with Spinner("Downloading data..."):
    # Code to download data goes here
```

This code will display a spinner icon beside the message "Downloading data..." while the data is being downloaded. Once the data has been downloaded, the spinner icon and message will disappear, indicating that the operation is complete.