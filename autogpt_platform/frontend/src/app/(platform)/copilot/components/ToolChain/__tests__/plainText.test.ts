import { describe, expect, it } from "vitest";
import { looksTechnical } from "../plainText";

describe("looksTechnical", () => {
  it.each([
    "Patching brief.md with sed",
    "Editing ~/workspace/dewleaf/brief",
    "Reading workspace/dewleaf/notes",
    "Checking reports/brief",
    "Running `npm install`",
    "Calling run_capability",
    "Saving 2616d1f1-16bb-43c0-9f0e-6a7c2a9d0b11",
    "cd dewleaf && ls",
    "git status",
    "Running git status",
    "Checking git log for changes",
    "cat brief",
    "cat notes",
    "head -n 5 calendar",
    "Counting rows with wc -l orders",
    "python3 report",
    "Running cat brief",
    "Ran head calendar",
    "Now running python3 report",
    "Executing jq on the export",
    "Calling node build",
    "Running make build",
    "Running sort orders",
    "Reading index.js",
    "Updating app.ts",
    "npm install",
  ])("flags %j", (text) => {
    expect(looksTechnical(text)).toBe(true);
  });

  it.each([
    "Updating your launch brief",
    "Checking the content calendar dates",
    "Planning the 24/7 support rota",
    "Posting on Instagram/Shopify and/or email",
    "Pricing at Rs 899 for 30ml",
    "Drafting Instagram posts",
    "Updating the cat food listings",
    "Checking the head office rota",
    "Writing the status update",
    "Running the cat food promo",
    "Ran the head office rota",
    "Running a check on the cat food stock",
    "Running sort on items",
    "Running export of orders",
    "Running find for duplicates",
    "Calling make for the launch",
    "Asking for a yes/no on the budget",
    "Weighing the pros/cons of each channel",
    "Mapping the input/output of the funnel",
    "Checking the Next.js storefront",
    "Updating the Node.js and Vue.js notes",
  ])("passes %j", (text) => {
    expect(looksTechnical(text)).toBe(false);
  });
});
