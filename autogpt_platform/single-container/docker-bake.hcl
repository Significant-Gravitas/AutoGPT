group "default" {
  targets = ["single-container"]
}

target "backend-server-base" {
  context    = "."
  dockerfile = "autogpt_platform/backend/Dockerfile"
  target     = "server-base"
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
  contexts = {
    autogpt-backend-base = "target:backend-server-base"
    autogpt-backend      = "target:backend-server"
  }
  tags = ["autogpt-platform:single-container-dev"]
}
