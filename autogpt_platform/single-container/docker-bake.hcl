group "default" {
  targets = ["single-container"]
}

target "backend-server" {
  context    = "."
  dockerfile = "autogpt_platform/backend/Dockerfile"
  target     = "server"
}

target "single-container" {
  context    = "."
  dockerfile = "autogpt_platform/single-container/Dockerfile"
  target     = "single-container"
  args = {
    NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true"
    NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS = "true"
    NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY = "true"
    NEXT_PUBLIC_FORCE_FLAG_ARTIFACTS = "true"
    NEXT_PUBLIC_FORCE_FLAG_ARTIFACTS_PAGE = "true"
    NEXT_PUBLIC_FORCE_FLAG_CHAT_WORKSPACE_FILES = "true"
    NEXT_PUBLIC_FORCE_FLAG_CHAT_SHARING = "true"
  }
  contexts = {
    autogpt-backend = "target:backend-server"
  }
  tags = ["autogpt-platform:single-container-dev"]
}
