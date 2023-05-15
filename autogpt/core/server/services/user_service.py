async def get_all_users():
    return {"message": "get_all_users has been run"}


async def get_user(user_id):
    return {"message": f"get_user has been run with userId: {user_id}"}


async def create_user():
    return {"message": "create_user has been run"}


async def update_user(user_id):
    return {"message": f"update_user has been run with userId: {user_id}"}


async def delete_user(user_id):
    return {"message": f"delete_user has been run with userId: {user_id}"}
