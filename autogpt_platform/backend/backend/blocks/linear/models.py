from pydantic import BaseModel


class Comment(BaseModel):
    id: str
    body: str


class CreateCommentInput(BaseModel):
    body: str
    issueId: str


class CreateCommentResponse(BaseModel):
    success: bool
    comment: Comment


class CreateCommentResponseWrapper(BaseModel):
    commentCreate: CreateCommentResponse


class Issue(BaseModel):
    id: str
    identifier: str
    title: str
    description: str | None
    priority: int


class CreateIssueResponse(BaseModel):
    issue: Issue


class Project(BaseModel):
    id: str
    name: str
    description: str
    priority: int
    progress: int
    content: str
