// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "AutoGPTMobileCore",
  platforms: [.macOS(.v13), .iOS(.v16)],
  products: [.library(name: "AutoGPTMobileCore", targets: ["AutoGPTMobileCore"])],
  targets: [
    .target(name: "AutoGPTMobileCore"),
    .testTarget(name: "AutoGPTMobileCoreTests", dependencies: ["AutoGPTMobileCore"]),
  ]
)
