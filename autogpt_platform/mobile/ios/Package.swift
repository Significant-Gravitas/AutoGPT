// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "AutoGPTShell",
  platforms: [.macOS(.v13), .iOS("17.4")],
  products: [.library(name: "ShellCore", targets: ["ShellCore"])],
  targets: [
    .target(name: "ShellCore", path: "AutoGPT/Core"),
    .testTarget(name: "ShellCoreTests", dependencies: ["ShellCore"], path: "Tests"),
  ]
)
