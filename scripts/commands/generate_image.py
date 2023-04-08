from operator import itemgetter
from commands import Command
from image_gen import generate_image

class Generate_Image(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        prompt = itemgetter('prompt')(kwargs)
        
        return generate_image(prompt)