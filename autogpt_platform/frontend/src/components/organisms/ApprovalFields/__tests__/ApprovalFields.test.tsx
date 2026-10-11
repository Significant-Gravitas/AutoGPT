import { expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { ApprovalFields } from "../ApprovalFields";

test.each([
  ["seconds", 59, "59 seconds"],
  ["seconds", 90, "1 minute 30 seconds"],
  ["seconds", 3_599, "59 minutes 59 seconds"],
  ["seconds", 5_400, "1 hour 30 minutes"],
  ["seconds", 86_400, "1 day"],
  [
    "cron",
    ["0 9 * * *", "0 17 * * *"],
    "Every day at 09:00, Every day at 17:00",
  ],
  // Anything the card cannot word shows as it did before hints existed.
  ["seconds", "soon", "soon"],
  ["cron", "at nine", "at nine"],
  ["date-time", "2026-10-02T09:00:00Z", "2026-10-02T09:00:00Z"],
  [null, 180, "180"],
])("format %s shows %j as %s", (format, value, text) => {
  render(
    <ApprovalFields
      fields={[{ key: "when", label: "When", format }]}
      values={{ when: value }}
    />,
  );
  expect(screen.getByText(text)).toBeDefined();
});
