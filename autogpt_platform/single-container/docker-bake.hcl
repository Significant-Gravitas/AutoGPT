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
  contexts = {
    autogpt-backend = "target:backend-server"
  }
  tags = ["autogpt-platform:single-container-dev"]
}
