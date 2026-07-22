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

  it("renders the admin sidebar navigation", () => {
    render(
      <AdminLayout>
        <div />
      </AdminLayout>,
    );
    expect(screen.getByText("Marketplace Management")).toBeDefined();
    expect(screen.getByText("System Diagnostics")).toBeDefined();
  });
});
