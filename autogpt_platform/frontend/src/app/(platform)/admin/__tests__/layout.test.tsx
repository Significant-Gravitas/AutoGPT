import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
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
});
