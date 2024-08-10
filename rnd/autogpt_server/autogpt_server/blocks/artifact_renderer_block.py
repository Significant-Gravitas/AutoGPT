from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField
import re
import base64
import markdown

class ArtifactRendererBlock(Block):
    class Input(BlockSchema):
        artifact_string: str = SchemaField(description="The input string containing an AGPT artifact to be rendered.")

    class Output(BlockSchema):
        artifact_data: dict = SchemaField(description="Processed artifact data for frontend rendering.")

    def __init__(self):
        super().__init__(
            id="7a8b9c0d-1e2f-3g4h-5i6j-7k8l9m0n1o2p",
            description="Processes an AGPT artifact for visual rendering within the block.",
            categories={BlockCategory.TEXT, BlockCategory.BASIC},
            input_schema=ArtifactRendererBlock.Input,
            output_schema=ArtifactRendererBlock.Output,
        )
        
    def parse_artifact(self, artifact_string):
        pattern = r'<agptArtifact\s+(.*?)>(.*?)</agptArtifact>'
        match = re.search(pattern, artifact_string, re.DOTALL)
        if match:
            attributes = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
            content = match.group(2).strip()
            return attributes, content
        return None, None

    def process_artifact(self, attributes, content):
        artifact_type = attributes.get('type', '')
        title = attributes.get('title', 'Untitled Artifact')
        identifier = attributes.get('identifier', '')
        language = attributes.get('language', '')

        processed_data = {
            'type': artifact_type,
            'title': title,
            'identifier': identifier,
            'language': language,
            'content': content
        }

        if artifact_type.startswith('image/'):
            processed_data['content'] = f"data:{artifact_type};base64,{content}"
        elif artifact_type == 'text/markdown':
            # Send markdown as plain text, don't convert to HTML
            processed_data['content'] = content
        elif artifact_type == 'application/vnd.agpt.code':
            # Keep the content as is for code snippets
            pass
        elif artifact_type == 'text/html' or artifact_type == 'image/svg+xml':
            # Keep HTML and SVG content as is
            pass
        else:
            processed_data['content'] = content

        return processed_data

    def run(self, input_data: Input) -> BlockOutput:
        attributes, content = self.parse_artifact(input_data.artifact_string)
        if attributes and content:
            processed_data = self.process_artifact(attributes, content)
            yield "artifact_data", processed_data
        else:
            yield "artifact_data", {"error": "Invalid artifact format"}