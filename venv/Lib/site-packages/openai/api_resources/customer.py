from openai.openai_object import OpenAIObject


class Customer(OpenAIObject):
    @classmethod
    def get_url(cls, customer, endpoint):
        return f"/customer/{customer}/{endpoint}"

    @classmethod
    def create(cls, customer, endpoint, **params):
        instance = cls()
        return instance.request("post", cls.get_url(customer, endpoint), params)

    @classmethod
    def acreate(cls, customer, endpoint, **params):
        instance = cls()
        return instance.arequest("post", cls.get_url(customer, endpoint), params)
