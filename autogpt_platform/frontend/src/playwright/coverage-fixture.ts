import { test as base } from "@playwright/test";
import { markChangelogSeen } from "./utils/changelog";
import { addCoverageReport } from "monocart-reporter";
import { readFile } from "node:fs/promises";
import path from "node:path";

const test = base.extend<{ autoTestFixture: void; changelogSeen: void }>({
  autoTestFixture: [
    async ({ page }, use) => {
      let hasCoverage = false;
      try {
        await page.coverage.startJSCoverage({ resetOnNavigation: false });
        hasCoverage = true;
      } catch {
        // coverage API not available (e.g. non-browser tests)
      }

      await use();

      if (hasCoverage) {
        try {
          const jsCoverageList = await page.coverage.stopJSCoverage();
          if (jsCoverageList.length > 0) {
            await addCoverageReport(jsCoverageList, test.info());
          }
        } catch {
          // Don't let coverage teardown failures mask real test failures
        }
      }
    },
    { scope: "test", auto: true },
  ],
  changelogSeen: [
    async ({ context, baseURL }, use) => {
      let markdown: string | undefined;
      if (process.env.PLAYWRIGHT_CHANGELOG_FIXTURE === "true") {
        const hostname = new URL(baseURL ?? "").hostname;
        if (!["localhost", "127.0.0.1", "[::1]"].includes(hostname)) {
          throw new Error("Changelog fixtures require a local test app");
        }
        markdown = await readFile(
          path.join(__dirname, "fixtures/changelog.md"),
          "utf8",
        );
      }
      await markChangelogSeen(context, markdown);
      await use();
    },
    { scope: "test", auto: true },
  ],
});

export { test };
export { expect } from "@playwright/test";
