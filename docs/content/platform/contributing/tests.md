# Testing

We use [Playwright](https://playwright.dev/) for our testing framework.

## Before you start

Almost all of the tests require that you are running the frontend and backend servers. You will hit strange and hard to debug errors if you don't have them running because the tests will try to interact with the application when it's not running in an interactable state.

## Running the tests

To run the tests, you can use the following commands:

Running the tests without the UI, and headless:

```bash
yarn test
```

If you want to run the tests in a UI where you can identify each locator used you can use the following command:

```bash
yarn test-ui
```

You can also pass `--debug` to the test command to open the browsers in view mode rather than headless. This works with both the `yarn test` and `yarn test-ui` commands.

```bash
yarn test --debug
```

In CI, we run the tests in headless mode, with multiple browsers, and retry a failed test up to 2 times.

You can find the full configuration in [playwright.config.ts](https://github.com/Significant-Gravitas/Autogpt/blob/master/autogpt_platform/frontend/playwright.config.ts).

### Debugging tests

There's a lot of different ways to debug tests.

My preferred is a mix of playwright's test editor and vscode.

No matter what you do, you should **always** double check that your locators are correct. Playwright will often "time out" and not give you the error message that the locator is incorrect because it can't find the element. You can do this via devtools on your browser and they should be visible on the elements tab when you use the inspect and select elements tools.

#### Using the playwright test editor

If you need to debug a test, you can use the below command to open the test in the playwright test editor. This is helpful if you want to see the test in the browser and see the state of the page as the test sees it and the locators it uses.

```bash
yarn test --debug --test-name-pattern="test-name"
```

#### Using vscode

You can install the [Playwright Test for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-playwright.playwright) extension to get autocomplete for the playwright api (id: `ms-playwright.playwright`).

Installing this will enable the `Test Explorer` view in vscode which allows you to run, debug, and view all tests in the current project. Adding breakpoints to your tests and running them will automatically open the test editor with the correct context.

## Setting up for generating tests

With playwright, you can generate tests from existing recordings of user sessions. This is useful for creating tests that are more representative of how a user would interact with the application. We generally use this for checking what ids stuff will have and what needs ids to be added.

It is super annoying to continuously login so I highly recommend using a saved session for your tests.
This will save a file called `.auth/gentest-user.json` that can be loaded for all future gentests so that you don't have to login every time.

### Saving a session for gen tests to always use

```bash
yarn gentests --save-storage .auth/gentest-user.json
```

Stop your session with `CTRL + C` after you are logged in and swap the `--save-storage` flag with `--load-storage` to load the session for all future tests.

### Loading a session for gen tests to always use

```bash
yarn gentests --load-storage .auth/gentest-user.json
```

## How to make a new test

Tests are composed of page objects and test files.

A page object is a class that contains methods for interacting with a page.

A test file is a file that contains tests for a page or a set of pages.

### Making a new Page Object

For tests, we use the [page object model](https://playwright.dev/docs/pom). This is a pattern where each page is a class that contains all the methods and locators for that page.
This is useful for keeping your tests organized and easy to read as well as ensuring that your tests only need to be updated in one place when the UI changes.

You should make a new page object (only when needing to add a new page, or **UI element** that is across multiple tests) using the following example.

We extend the `BasePage` class which contains shared methods for pages that have the common functionality like a navbar. If you add something like that (for example a sidebar) you should add it to the `BasePage` class. Otherwise, you should make a new page object.

Each page object should be in its own file and be named like `page-name.page.ts`.
A page object should contain methods that are actions that a user can do on that page. For example, clicking a button, filling out a form, etc. It should also contain the various helpful abstractions that are unique to that page. For example, the `BuildPage` has a method to connect blocks together.

This is a shortened example of a page object for the profile page:

<!-- I know there's a floating } but it closes the imported code block and makes this a valid copy-able block -->

```typescript title="frontend/src/tests/pages/profile.page.ts"
--8<-- "autogpt_platform/frontend/src/tests/pages/profile.page.ts:ProfilePageExample"
}
```

### Making a new Test File

For tests, we use our page objects to create tests. Each test file should be in the `tests` folder and be named like `test-name.spec.ts`. A test file can contain multiple tests. Each of which shuld be related to the same conceptual function. For example, a test file for the build page could have tests for building agents, creating inputs and outputs, and connecting blocks. If the you wanted to speciifically test building agents, you could make a new test called `building-agents.spec.ts`.

Tests can inherit from one or more page objects, have pre-actions, and have post-actions, as well as many other features. You can learn more about the different features and how to use them [here](https://playwright.dev/docs/test-actions).

A good focused (`unit` or `single concept`) test will:

- Have a short name that describes what it is testing
- Have a single concept (building a agent, adding all blocks, connecting two blocks, etc.)
- Check pre-conditions, actions, and post-conditions, as well as have multiple validations along the way

A good non-focused (`integration` or `multiple concepts`) test will:

- Have a short name that describes what it is testing
- Have multiple concepts (building agents, creating-?exporting->importing->running an agent, connecting blocks in multiple ways with multiple inputs and outputs, etc.)
- Have a clear user experience that they are making sure works (for example, clicking the build button and making sure the agent is built, or clicking the export button and making sure the agent is exported and shows up in the monitoring system)
- Not focus on a single concept, but instead test the flow of the application as a whole. Remember you're not testing the pixel perfect UI, but the user experience.

A good test suite will have a healthy mix of focused and non-focused tests.

### Example Focused Test & Explanation

```typescript title="frontend/src/tests/build.spec.ts"
--8<-- "autogpt_platform/frontend/src/tests/build.spec.ts:BuildPageExample"
});
```

1. The `test.describe` is used to group tests together. In this case, it's used to group all the tests for the build page together.
2. The `let buildPage: BuildPage;` is used to create a new instance of the build page.
3. The `test.beforeEach` is used to run code before each test. In this case, it's used to login the user before each test. `page` is the page object that is passed in from the fixture, `loginPage` is the page object for the login page, and `testUser` is the user object that is passed in from the fixture. The fixture is used to handle authentication and other common shared state tasks.
4. The `await page.goto("/login");` is used to navigate to the login page.
5. The `await test.expect(page).toHaveURL("/");` is used to check that the page has navigated to the home page (and are therefore logged in).
6. The `test("user can add a block", async ({ page }) => {` is used to define a new test.
7. The `await test.expect(buildPage.isLoaded()).resolves.toBeTruthy();` is used to check that the build page has loaded. This could reasonably done in the `test.beforeEach` but is done here for clarity due to other tests in this suite.
8. The `await test.expect(page).toHaveURL(new RegExp("/.*build"));` is used to check that the page has navigated to the build page.
9. The `await buildPage.closeTutorial();` is used to close the tutorial on the build page, noticibly this wrapping funciton doesn't actually care if its open or not, it ensures that it **will** be closed. This is a useful and common pattern for ensuring that something will be done, without caring if it is already done. It could be used for things like toggling a setting, closing/opening a sidebar, etc.
10. The `await buildPage.openBlocksPanel();` is used to open the blocks panel on the build page, in the same way described for the `closeTutorial` function.
11. The `await buildPage.addBlock(block);` is used to add a specific block to the build page. It's another utility function that could be done in line, but due to how the Page Object pattern works, we should keep them in the page object. (It's also useful for keeping the test code cleaner and is used in other tests)
12. The `await buildPage.closeBlocksPanel();` is used to close the blocks panel on the build page.
13. The `await test.expect(buildPage.hasBlock(block)).resolves.toBeTruthy();` is used to check that the block has been added to the build page.

### Passing information between tests

You can pass information between tests using the `testInfo` object. This is useful for things like passing the id of an agent between beforeAll so that you can have a shared setup for multiple tests.

```typescript title="frontend/src/tests/monitor.spec.ts"
--8<-- "autogpt_platform/frontend/src/tests/monitor.spec.ts:AttachAgentId"

  test("test can read the agent id", async ({ page }, testInfo) => {
    --8<-- "autogpt_platform/frontend/src/tests/monitor.spec.ts:ReadAgentId"
    /// ... Do something with the agent id here
  });
});
```

## See Also

- [Writing Tests](https://playwright.dev/docs/writing-tests)
- [Code Generation](https://playwright.dev/docs/codegen-intro)
- [Test UI Mode](https://playwright.dev/docs/test-ui-mode)
- [Trace Viewer](https://playwright.dev/docs/trace-viewer-intro)
- [Getting Started with VSCode](https://playwright.dev/docs/getting-started-vscode)
- [Debugging Tests](https://playwright.dev/docs/debug)
- [Test Fixtures](https://playwright.dev/docs/test-fixtures)
- [Global Setup and Teardown](https://playwright.dev/docs/test-global-setup-teardown)
- [Test Parameterization](https://playwright.dev/docs/test-parameterize)
- [Test Events](https://playwright.dev/docs/events)
- [Test Components](https://playwright.dev/docs/test-components)
- [Test Sharding](https://playwright.dev/docs/test-sharding)
- [Accessibility Testing](https://playwright.dev/docs/accessibility-testing)
- [Authentication](https://playwright.dev/docs/auth)
- [Mocking](https://playwright.dev/docs/mock)
- [Mock Browser APIs](https://playwright.dev/docs/mock-browser-apis)
- [Code Generation](https://playwright.dev/docs/codegen)
- [Pages](https://playwright.dev/docs/pages)
- [Test Annotations](https://playwright.dev/docs/test-annotations)
