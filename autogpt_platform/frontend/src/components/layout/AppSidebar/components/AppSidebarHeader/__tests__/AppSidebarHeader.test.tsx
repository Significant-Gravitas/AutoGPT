import { render, screen } from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { AppSidebarHeader } from "../AppSidebarHeader";

function renderHeader() {
  return render(
    <SidebarProvider>
      <AppSidebarHeader />
    </SidebarProvider>,
  );
}

describe("AppSidebarHeader", () => {
  it("renders the AutoGPT home link pointing at /copilot", () => {
    renderHeader();
    const homeLink = screen.getByRole("link", { name: "AutoGPT" });
    expect(homeLink.getAttribute("href")).toBe("/copilot");
  });

  it("shows the 'Collapse sidebar' control while expanded", () => {
    renderHeader();
    expect(
      screen.getByRole("button", { name: "Collapse sidebar" }),
    ).toBeDefined();
  });

  it("toggles to 'Expand sidebar' after collapsing", async () => {
    const user = userEvent.setup();
    renderHeader();

    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));

    expect(
      await screen.findByRole("button", { name: "Expand sidebar" }),
    ).toBeDefined();
  });
});
