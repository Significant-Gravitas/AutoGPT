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
    let initial = XCTAttachment(screenshot: app.screenshot())
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
    let scrolled = XCTAttachment(screenshot: app.screenshot())
    scrolled.name = "Native status - actions reached after scrolling"
    scrolled.lifetime = .keepAlways
    add(scrolled)
  }

  @MainActor
  func testStartsHostedChatAndOffersConnectionSettings() {
    let app = XCUIApplication()
    app.launch()
    XCTAssertTrue(app.navigationBars["AutoGPT"].waitForExistence(timeout: 10))
    app.buttons["App menu"].tap()
    app.buttons["Server settings"].tap()
    XCTAssertTrue(app.alerts["Server settings"].waitForExistence(timeout: 3))
    XCTAssertTrue(app.textFields["Server address"].exists)
    app.buttons["Cancel"].tap()
  }
}
