# Testing

The AutoGPT Platform uses several test frameworks at different layers:

- Backend tests use pytest.
- Frontend integration tests use Vitest, React Testing Library, and MSW. These
  are the primary frontend tests.
- End-to-end browser tests use [Playwright](https://playwright.dev/).
- Design system components use Storybook stories for visual coverage.

Run these Bash commands from the repository root after installing dependencies:

Backend tests require Docker running and `autogpt_platform/.env` (copy from
`.env.default`). Playwright requires the backend running; `pnpm test` builds
and starts the frontend, reusing an existing server when available.

```bash
(cd autogpt_platform/backend && poetry run test)
(cd autogpt_platform/autogpt_libs && poetry run pytest)
(cd autogpt_platform/frontend && pnpm test:unit) # no servers required
(cd autogpt_platform/frontend && pnpm test) # E2E: start the backend first
```

For the frontend's fast edit/test loop, run `pnpm test:unit:watch` from its workspace.

See the backend [testing guide](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/autogpt_platform/backend/TESTING.md)
and frontend [testing guide](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/autogpt_platform/frontend/TESTING.md)
for patterns, commands, and guidance on choosing integration or end-to-end
coverage.

## Before you start

Playwright tests require the backend server. `pnpm test` builds and starts the
frontend for you, or reuses an existing server. Wait for the backend to be ready
before running browser tests.

## Running the Playwright tests

To run the tests, you can use the following commands:

Running the tests without the UI, and headless:

```bash
pnpm test
```

If you want to run the tests in a UI where you can identify each locator used you can use the following command:

```bash
pnpm test-ui
```

You can also pass `--debug` to the test command to open the browsers in view mode rather than headless. This works with both the `pnpm test` and `pnpm test-ui` commands.

```bash
pnpm test --debug
```

In CI, the [full-stack workflow](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/.github/workflows/platform-fullstack-ci.yml)
runs the Chromium project headlessly with `--retries=0 --trace=retain-on-failure`,
overriding the shared config's CI defaults. Its JSON validator requires every test to have exactly one successful
attempt and rejects skipped, flaky, unexpected, or missing results and top-level
errors. See the shared settings in
[playwright.config.ts](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/autogpt_platform/frontend/playwright.config.ts).

Failure traces and test artifacts may contain requests, cookies, and session
state. Use isolated test accounts, never production accounts, and avoid exposing
credentials in traced requests or application output.

## Continuous integration

### Sharding

The [backend workflow](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/.github/workflows/platform-backend-ci.yml)
and [frontend workflow](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/.github/workflows/platform-frontend-ci.yml)
define the shard matrix. CI divides the backend and frontend integration suites into disjoint parallel
shards. Backend shards select data, copilot, and util/executor paths; the
remainder uses normal workspace discovery excluding those paths, including new
top-level test files and directories. All shards explicitly load the backend's
pytest configuration. Backend shards use real PostgreSQL, RabbitMQ, and Redis,
with an isolated database, virtual host, and Redis cluster for each shard.
Each shard pays its own service-startup cost, so adding shards also multiplies setup work.

### Caching

Dependency and browser caches are keyed to their lockfiles. Browser caches can
fall back to a previous cache for the same runner OS, then install missing versions.
Docker builds reuse unchanged layers. Full-stack cache export is limited to trusted `dev` pushes or an
explicit cache-publishing dispatch. Single-container builds use GHCR caches per
architecture instead of the repository Actions cache, except on pull requests,
which do not read or write registry caches. Manual dispatches read and update a
branch-specific cache, with the trusted `dev` cache as a read-only fallback.
Only a push to `dev` writes the trusted cache used by release builds.
The generated E2E seed-data cache is keyed to the exact commit under test. Warm
caches may avoid repeated dependency downloads, seed generation, or cache
export, but they never bypass tests, linting, type checks, coverage collection, image builds, image
smoke tests, or security scans.

### Test reports

Every test job in the validation workflows uploads a JUnit XML or JSON report,
plus coverage artifacts where applicable; this does not describe release-only
publication jobs. The [JUnit validator](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/.github/scripts/validate_junit.py)
and [Playwright validator](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/.github/scripts/validate_playwright_json.py)
fail validation on missing or malformed reports,
zero discovered tests, count mismatches, failures, and errors. Frontend
integration and Playwright reports also reject skipped tests; the
[single-container appliance workflow](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/.github/workflows/platform-single-container-docker.yml)
allows one exact test ID to skip when Bash 3 is unavailable. The
[unittest reporter](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/.github/scripts/run_unittest_junit.py)
enforces this policy, including class, module, and subtest skips. Backend skips must match exact test IDs in
`.github/scripts/backend-allowed-skips.json`, seeded from the 136 existing skips
per Python version in run 33918906861; a new skip fails validation. Do not add an
ID merely to make CI green: investigate and review the changed skip policy.
An entirely skipped backend shard also fails.
Wrapped commands and malformed backend/frontend reports produce a synthetic
machine-readable error. Single-container validation instead uploads its original
reports and leaves `status.json` marked `validated: false` if verification fails.

### Manual dispatch

These CI workflows have no required manual-dispatch inputs. Backend
dispatches can refresh a stacked PR's backend coverage with
`gh workflow run platform-backend-ci.yml --ref <branch> -f pr_number=<PR-number>`;
the optional `pr_number` only selects the Codecov upload target. Full-stack
dispatches import caches by default; set `publish_build_cache` only when a test
commit should deliberately refresh them. A manually dispatched single-container
run builds, smoke-tests, and scans the image but cannot publish it; publication
jobs are reachable only from a release event. Single-container validation also
runs on every `dev` push and release, while pull requests trigger it only when
appliance packaging inputs change.

### When CI fails

Open the failed Actions run and download its test-report and coverage artifacts
from the run summary. JUnit failures identify the test and traceback; validator
errors identify missing, malformed, or unexpected skipped results. Do not treat
a successful upload as proof the tests passed.

To reproduce a backend shard after configuring the same local services, run
`poetry run pytest -c pyproject.toml backend/copilot` from the backend workspace
(substitute the failing shard's paths). For a frontend shard, run
`pnpm test:unit --shard=1/4` from the frontend workspace, using its CI shard number.

### Debugging tests

There's a lot of different ways to debug tests.

My preferred is a mix of playwright's test editor and vscode.

No matter what you do, you should **always** double check that your locators are correct. Playwright will often "time out" and not give you the error message that the locator is incorrect because it can't find the element. You can do this via devtools on your browser and they should be visible on the elements tab when you use the inspect and select elements tools.

#### Using the playwright test editor

If you need to debug a test, you can use the below command to open the test in the playwright test editor. This is helpful if you want to see the test in the browser and see the state of the page as the test sees it and the locators it uses.

```bash
pnpm test --debug --test-name-pattern="test-name"
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
pnpm gentests --save-storage .auth/gentest-user.json
```

Stop your session with `CTRL + C` after you are logged in and swap the `--save-storage` flag with `--load-storage` to load the session for all future tests.

### Loading a session for gen tests to always use

```bash
pnpm gentests --load-storage .auth/gentest-user.json
```

## How to make a new test

Tests are composed of page objects and test files.

Do not commit skipped or todo Vitest/Playwright tests. Fix the prerequisite or
ask maintainers to review an explicit policy exception; do not hide a failing
test behind an environment check.

The current Playwright config discovers `*-happy-path.spec.ts` files under
`autogpt_platform/frontend/src/playwright/`. Other filenames are not collected;
follow that naming pattern for tests intended to run in CI.

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

```typescript title="frontend/src/playwright/pages/profile.page.ts"
--8<-- "autogpt_platform/frontend/src/playwright/pages/profile.page.ts:ProfilePageExample"
}
```

### Making a new Test File

For tests, we use our page objects to create tests. Each test file should be in `autogpt_platform/frontend/src/playwright/` and be named like `test-name-happy-path.spec.ts`. A test file can contain multiple tests. Each of which should be related to the same conceptual function. For example, a test file for the build page could have tests for building agents, creating inputs and outputs, and connecting blocks. If you wanted to specifically test building agents, you could make a new test called `building-agents-happy-path.spec.ts`.

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

### Example focused test

This uses the current coverage fixture and builder page object. Save new cases
under `autogpt_platform/frontend/src/playwright/` using the
`*-happy-path.spec.ts` filename pattern.

```typescript title="frontend/src/playwright/building-agents-happy-path.spec.ts"
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";

test.use({ storageState: E2E_AUTH_STATES.builder });

test("builder saves an agent", async ({ page }) => {
  const buildPage = new BuildPage(page);
  await buildPage.createAndSaveSimpleAgent("Example Agent");

  await expect(page).toHaveURL(/flowID=/);
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();
});
```

The coverage fixture preserves coverage collection. The storage state supplies
an isolated test account, the page object encapsulates UI interactions, and the
assertions verify both navigation and the saved agent's runnable state.
See the current
[builder tests](https://github.com/Significant-Gravitas/AutoGPT/blob/dev/autogpt_platform/frontend/src/playwright/builder-happy-path.spec.ts)
for full scenarios and timeout choices.

### Passing information within a test

Keep tests independent: do not make one test consume an ID created by another.
Use local variables or fixtures to share setup within a test, and use
`testInfo.attach` for safe diagnostics. Avoid attaching credentials or session
state. See Playwright's
[fixtures guide](https://playwright.dev/docs/test-fixtures)
and [TestInfo API](https://playwright.dev/docs/api/class-testinfo).

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
