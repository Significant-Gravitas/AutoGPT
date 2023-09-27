import uuid

from fastapi import Request


class UserService:
    def __init__(self):
        self.users = {}

    def get_user_id(self, request: Request) -> uuid.UUID:
        # TODO: something real.  I don't know how this works.
        hostname = request.client.host
        port = request.client.port
        user = f"{hostname}:{port}"
        if user not in self.users:
            self.users[user] = uuid.uuid4()
        return self.users[user]


USER_SERVICE = UserService()
