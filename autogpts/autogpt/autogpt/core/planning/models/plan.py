from pydantic import BaseModel
from autogpt.core.planning.models.tasks import Task

class Plan(BaseModel):
    tasks : list[Task]

    # FIXME !
    def dump(self, depth = 0) -> dict:
        plan = self.dict()
        return_dict = {}
        for task in plan.tasks : 
            return_dict = task.name