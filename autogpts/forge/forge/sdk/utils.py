import inspect
import sys
import traceback


def get_exception_message():
    """Get current exception type and message."""
    exc_type, exc_value, _ = sys.exc_info()
    exception_message = f"{exc_type.__name__}: {exc_value}"
    return exception_message


def get_detailed_traceback():
    """Get current exception traceback with local variables."""
    _, _, exc_tb = sys.exc_info()
    detailed_traceback = "Traceback (most recent call last):\n"
    formatted_tb = traceback.format_tb(exc_tb)
    detailed_traceback += "".join(formatted_tb)

    # Optionally add local variables to the traceback information
    detailed_traceback += "\nLocal variables by frame, innermost last:\n"
    while exc_tb:
        frame = exc_tb.tb_frame
        lineno = exc_tb.tb_lineno
        function_name = frame.f_code.co_name

        # Format frame information
        detailed_traceback += (
            f"  Frame {function_name} in {frame.f_code.co_filename} at line {lineno}\n"
        )

        # Get local variables for the frame
        local_vars = inspect.getargvalues(frame).locals
        for var_name, value in local_vars.items():
            detailed_traceback += f"    {var_name} = {value}\n"

        exc_tb = exc_tb.tb_next

    return detailed_traceback
