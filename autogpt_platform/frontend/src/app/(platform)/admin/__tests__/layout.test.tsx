import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { environment } from "@/services/environment";
import AdminLayout from "../layout";

const ADMIN_LINKS = [
  { text: "Marketplace Management", href: "/admin/marketplace" },
  { text: "User Spending", href: "/admin/spending" },
  { text: "System Diagnostics", href: "/admin/diagnostics" },
  { text: "User Impersonation", href: "/admin/impersonation" },
  { text: "Rate Limits", href: "/admin/rate-limits" },
  { text: "Platform Costs", href: "/admin/platform-costs" },
  { text: "Execution Analytics", href: "/admin/execution-analytics" },
  { text: "Bot Analytics", href: "/admin/bots" },
  { text: "Block Cost Estimates", href: "/admin/block-cost-estimates" },
  { text: "Memory Inspector", href: "/admin/memory" },
  { text: "Admin User Management", href: "/admin/settings" },
];

describe("AdminLayout", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders child content", () => {
    render(
      <AdminLayout>
        <div>Test Child</div>
      </AdminLayout>,
    );
    expect(screen.getByText("Test Child")).toBeDefined();
  });

  it("renders every admin sidebar link with the correct href", () => {
    render(
      <AdminLayout>
        <div />
      </AdminLayout>,
    );
    const links = screen.getAllByRole("link");
    for (const { text, href } of ADMIN_LINKS) {
      const link = links.find((el) => el.getAttribute("href") === href);
      expect(link).toBeDefined();
      expect(link?.textContent).toContain(text);
    }
  });

  it("shows the Test Data link on a local stack", () => {
    vi.spyOn(environment, "isLocal").mockReturnValue(true);
    render(
      <AdminLayout>
        <div />
      </AdminLayout>,
    );
    const link = screen
      .getAllByRole("link")
      .find((el) => el.getAttribute("href") === "/admin/test-data");
    expect(link).toBeDefined();
  });

  it("hides the Test Data link outside a local stack", () => {
    vi.spyOn(environment, "isLocal").mockReturnValue(false);
    render(
      <AdminLayout>
        <div />
      </AdminLayout>,
    );
    const link = screen
      .getAllByRole("link")
      .find((el) => el.getAttribute("href") === "/admin/test-data");
    expect(link).toBeUndefined();
  });
});
