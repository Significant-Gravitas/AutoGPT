import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import AdminLayout from "../layout";

describe("AdminLayout", () => {
  it("renders child content", () => {
    render(
      <AdminLayout>
        <div>Test Child</div>
      </AdminLayout>,
    );
    expect(screen.getByText("Test Child")).toBeDefined();
  });

  it("does not render its own sidebar (uses the shared app sidebar)", () => {
    render(
      <AdminLayout>
        <div />
      </AdminLayout>,
    );
    expect(screen.queryByTestId("sidebar")).toBeNull();
  });
});
