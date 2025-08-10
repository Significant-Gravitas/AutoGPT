import { chromium, webkit } from "@playwright/test";

export const getBrowser = async () => {
  const browserType = process.env.BROWSER_TYPE || "chromium";

  const browser =
    browserType === "webkit"
      ? await webkit.launch({ headless: true })
      : await chromium.launch({ headless: true });

  return browser;
};
