import { test as base } from "@playwright/test";
import { addCoverageReport } from "monocart-reporter";

const test = base.extend({
  autoTestFixture: [
    async ({ page }, use) => {
      await page.coverage.startJSCoverage({ resetOnNavigation: false });

      await use("");

      const jsCoverageList = await page.coverage.stopJSCoverage();
      await addCoverageReport(jsCoverageList, test.info());
    },
    { scope: "test", auto: true },
  ],
});

export { test };
export { expect } from "@playwright/test";
