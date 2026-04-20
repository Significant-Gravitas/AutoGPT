import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import AdminLayout from "../layout";

vi.mock("@/components/__legacy__/Sidebar", () => ({
  Sidebar: ({
    linkGroups,
  }: {
    linkGroups: { links: { text: string }[] }[];
  }) => (
    <nav data-testid="sidebar">
      {linkGroups[0].links.map((link) => (
        <span key={link.text}>{link.text}</span>
      ))}
    </nav>
  ),
}));

describe("AdminLayout", () => {
  it("renders sidebar with System Diagnostics link", () => {
    render(
      <AdminLayout>
        <div>Child Content</div>
      </AdminLayout>,
    );
    expect(screen.getByText("System Diagnostics")).toBeDefined();
  });

  it("renders child content", () => {
    render(
      <AdminLayout>
        <div>Test Child</div>
      </AdminLayout>,
    );
    expect(screen.getByText("Test Child")).toBeDefined();
  });

  it("renders all admin navigation links", () => {
    render(
      <AdminLayout>
        <div />
      </AdminLayout>,
    );
    expect(screen.getByText("Marketplace Management")).toBeDefined();
    expect(screen.getByText("User Spending")).toBeDefined();
    expect(screen.getByText("System Diagnostics")).toBeDefined();
    expect(screen.getByText("User Impersonation")).toBeDefined();
    expect(screen.getByText("Rate Limits")).toBeDefined();
    expect(screen.getByText("Platform Costs")).toBeDefined();
    expect(screen.getByText("Execution Analytics")).toBeDefined();
    expect(screen.getByText("Admin User Management")).toBeDefined();
  });
});
