from .agents import *

from fastapi import HTTPException
async def validate_request(request: Request, requestmodel :BaseModel ) -> BaseModel: 
    try : 
        body = await request.json()  
        request = requestmodel(**body)  
    except Exception as ve: 
            raise HTTPException(status_code=422, detail=str(ve))
    return 