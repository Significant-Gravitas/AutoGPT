import XCTest

final class AutoGPTUITests: XCTestCase {
  @MainActor
  func testLargeStatusActionsRemainReachableInLandscape() {
    let app = XCUIApplication()
    app.launchEnvironment["AUTOGPT_UI_TEST_SCREEN"] = "large-status"
    XCUIDevice.shared.orientation = .landscapeLeft
    defer { XCUIDevice.shared.orientation = .portrait }
    app.launch()

    let scroll = app.scrollViews["Native status"]
    XCTAssertTrue(scroll.waitForExistence(timeout: 5))
    XCTAssertTrue(app.staticTexts["Native status title"].exists)
    XCTAssertTrue(app.staticTexts["Native status message"].exists)
    let windowFrame = app.windows.firstMatch.frame
    XCTAssertGreaterThan(
      windowFrame.width, windowFrame.height, "The app window must be in landscape.")
    let initial = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
    initial.name = "Native status - landscape accessibility XXXL"
    initial.lifetime = .keepAlways
    add(initial)
    for title in ["Sign in to AutoGPT", "Open in browser"] {
      let button = app.buttons[title]
      for _ in 0..<6 {
        if button.isHittable { break }
        scroll.swipeUp()
      }
      XCTAssertTrue(button.exists, "The \(title) action must remain accessible.")
      XCTAssertTrue(button.isHittable, "The \(title) action must be reachable by scrolling.")
    }
    let scrolled = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
    scrolled.name = "Native status - actions reached after scrolling"
    scrolled.lifetime = .keepAlways
    add(scrolled)
  }

  @MainActor
  func testSignInAndConnectionSettingsUseNativeLayout() {
    let app = XCUIApplication()
    app.launchEnvironment["AUTOGPT_UI_TEST_SCREEN"] = "sign-in"
    app.launch()
    XCTAssertTrue(app.navigationBars["AutoGPT"].waitForExistence(timeout: 10))
    XCTAssertTrue(app.staticTexts["Welcome back"].exists)
    capture("Native sign-in - portrait")
    app.buttons["App menu"].tap()
    app.buttons["Server settings"].tap()
    XCTAssertTrue(app.staticTexts["Server settings title"].waitForExistence(timeout: 3))
    XCTAssertTrue(app.textFields["Server address"].exists)
    capture("Server settings - portrait")
    app.buttons["Cancel"].tap()
  }

  @MainActor
  func testLargeServerSettingsRemainReachableInLandscape() {
    let app = XCUIApplication()
    app.launchEnvironment["AUTOGPT_UI_TEST_SCREEN"] = "large-status"
    XCUIDevice.shared.orientation = .landscapeLeft
    defer { XCUIDevice.shared.orientation = .portrait }
    app.launch()
    app.buttons["App menu"].tap()
    app.buttons["Server settings"].tap()
    let scroll = app.scrollViews["Server settings content"]
    XCTAssertTrue(scroll.waitForExistence(timeout: 5))
    XCTAssertTrue(app.staticTexts["Server settings title"].exists)
    let field = app.textFields["Server address"]
    for _ in 0..<8 {
      if field.isHittable { break }
      scroll.swipeUp()
    }
    XCTAssertTrue(field.isHittable)
    field.tap()
    XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 3))
    capture("Server settings - landscape accessibility XXXL with keyboard")
    for title in ["Connect", "Cancel"] {
      let button = app.buttons[title]
      for _ in 0..<8 {
        if button.isHittable { break }
        scroll.swipeUp()
      }
      XCTAssertTrue(button.isHittable, "The \(title) action must be reachable by scrolling.")
    }
    app.buttons["Cancel"].tap()
    XCTAssertTrue(app.buttons["App menu"].waitForExistence(timeout: 3))
  }

  @MainActor
  private func capture(_ name: String) {
    let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
    attachment.name = name
    attachment.lifetime = .keepAlways
    add(attachment)
  }
}
