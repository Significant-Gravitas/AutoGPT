import { test as base } from "@playwright/test";
import { addCoverageReport } from "monocart-reporter";

const test = base.extend<{ autoTestFixture: void }>({
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
});

export { test };
export { expect } from "@playwright/test";
