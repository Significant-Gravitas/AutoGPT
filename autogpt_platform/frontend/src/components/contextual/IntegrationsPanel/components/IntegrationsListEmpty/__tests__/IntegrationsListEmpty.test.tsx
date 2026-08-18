import { describe, expect, test } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";

import { IntegrationsListEmpty } from "../IntegrationsListEmpty";

describe("IntegrationsListEmpty", () => {
  test("renders connect-a-service copy when there is no active query", () => {
    render(<IntegrationsListEmpty query="" />);
    expect(screen.getByText("No integration connected")).toBeDefined();
    expect(
      screen.getByText(
        /Connect a service to let your agents use third-party tools/i,
      ),
    ).toBeDefined();
  });

  test("treats whitespace-only queries as no query", () => {
    render(<IntegrationsListEmpty query="   " />);
    expect(screen.getByText("No integration connected")).toBeDefined();
  });

  test("renders no-results copy and echoes the trimmed query", () => {
    render(<IntegrationsListEmpty query="  notion  " />);
    expect(screen.getByText("No integrations found")).toBeDefined();
    expect(
      screen.getByText(/No integrations match "notion"\. Try a different/i),
    ).toBeDefined();
  });
});
