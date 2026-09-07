import XCTest

final class AutoGPTUITests: XCTestCase {
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
