from openai.api_resources.abstract import DeletableAPIResource, ListableAPIResource


class Model(ListableAPIResource, DeletableAPIResource):
    OBJECT_NAME = "models"
