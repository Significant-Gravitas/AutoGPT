import { test as base } from "@playwright/test";
import { addCoverageReport } from "monocart-reporter";

const test = base.extend<{ autoTestFixture: string }>({
  autoTestFixture: [
    async ({ page }, use) => {
      let hasCoverage = false;
      try {
        await page.coverage.startJSCoverage({ resetOnNavigation: false });
        hasCoverage = true;
      } catch {
        // coverage API not available (e.g. non-browser tests)
      }

      await use("");

      if (hasCoverage) {
        const jsCoverageList = await page.coverage.stopJSCoverage();
        if (jsCoverageList.length > 0) {
          await addCoverageReport(jsCoverageList, test.info());
        }
      }
    },
    { scope: "test", auto: true },
  ],
});

export { test };
export { expect } from "@playwright/test";
