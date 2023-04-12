## Function Description

The `clean_input()` function prompts the user for input and returns it as a string. If the user interrupts the function by pressing `ctrl+c` (KeyboardInterrupt), the function prints a message and exits the program.

## Parameters

- `prompt`: an optional parameter used as the prompt string for the input function. The default value is an empty string.

## Return Value

The `clean_input()` function returns a string value of the input given by the user.

## Example

```python
text = clean_input("Enter some text:")
print("You entered:", text)
```

Output:

```
Enter some text:Hello World!
You entered: Hello World!
```