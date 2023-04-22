import os

class PDFLogger:
    def __init__(self):
        # create log directory if it doesn't exist
        this_files_dir_path = os.path.dirname(__file__)
        log_dir = os.path.join(this_files_dir_path, "../logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.output_file = os.path.join(log_dir, "log.typ")
        
        # write to the file 
        
    def _write(self, content):
         with open(self.output_file, 'w') as file:  # Change 'a' to 'w' if you want to overwrite the file instead of appending
            file.write(content + '\n')  # Adding a newline character for better formatting
    
    def write_header(self, header):
        self._write(f"= {header}")        
    


logger = PDFLogger()
logger.write_header("Hello World")
logger.write_header("Hello Venus and Jupyter")


