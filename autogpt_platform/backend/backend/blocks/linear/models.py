from backend.sdk import BaseModel


class User(BaseModel):
    id: str
    name: str


class Comment(BaseModel):
    id: str
    body: str
    createdAt: str | None = None
    user: User | None = None


class CreateCommentInput(BaseModel):
    body: str
    issueId: str


class CreateCommentResponse(BaseModel):
    success: bool
    comment: Comment


class CreateCommentResponseWrapper(BaseModel):
    commentCreate: CreateCommentResponse


class Project(BaseModel):
    id: str
    name: str
    description: str | None = None
    priority: int | None = None
    progress: float | None = None
    content: str | None = None


class Issue(BaseModel):
    id: str
    identifier: str
    title: str
    description: str | None
    priority: int
    project: Project | None = None
    createdAt: str | None = None
    comments: list[Comment] | None = None
    assignee: User | None = None


class CreateIssueResponse(BaseModel):
    issue: Issue
