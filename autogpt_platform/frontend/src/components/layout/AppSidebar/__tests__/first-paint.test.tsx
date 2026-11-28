import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { getCurrentUser, validateSession } from "@/lib/auth/actions";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { isDrawnTransparent } from "@/tests/integrations/transparency";
import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { renderToString } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { AppSidebar } from "../AppSidebar";

vi.mock("@/lib/auth/actions", () => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  serverLogout: vi.fn(),
}));

vi.mock("../components/SidebarUserActions/SidebarUserActions", () => ({
  SidebarUserActions: () => null,
}));

function sidebar() {
  return (
    <QueryClientProvider client={getQueryClient()}>
      <BackendAPIProvider>
        <NuqsTestingAdapter>
          <SidebarProvider>
            <AppSidebar />
          </SidebarProvider>
        </NuqsTestingAdapter>
      </BackendAPIProvider>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  useAuthStore.setState(useAuthStore.getInitialState());
  getQueryClient().clear();
  vi.mocked(getCurrentUser).mockResolvedValue({ user: null });
  vi.mocked(validateSession).mockResolvedValue({ user: null, isValid: false });
});

afterEach(() => {
  document.body.innerHTML = "";
});

describe("AppSidebar first paint", () => {
  test("the server's HTML draws the sidebar's links, before the JavaScript has run", () => {
    const container = document.createElement("div");
    container.innerHTML = renderToString(sidebar());
    document.body.appendChild(container);

    const links = within(container).getAllByRole("link", { name: "Home" });
    expect(links.length).toBeGreaterThan(0);
    expect(links.filter(isDrawnTransparent)).toEqual([]);
  });

  test("mounted after the page has loaded, the links still enter", () => {
    render(sidebar());

    expect(
      isDrawnTransparent(screen.getAllByRole("link", { name: "Home" })[0]),
    ).toBe(true);
  });
});
