import { afterEach, describe, expect, test, vi } from "vitest";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { TourSidebar } from "../TourSidebar";

vi.mock("@/hooks/use-mobile", () => ({ useIsMobile: () => true }));

describe("TourSidebar cookie settings on a phone", () => {
  afterEach(() => {
    removeCookiebot();
    vi.unstubAllEnvs();
    vi.useRealTimers();
  });

  test("closes the sidebar sheet before opening the Cookiebot dialog", async () => {
    configureCookiebot();
    const { renew } = installCookiebot({ statistics: true });
    vi.useFakeTimers({ shouldAdvanceTime: true });

    render(
      <SidebarProvider>
        <SidebarTrigger />
        <TourSidebar />
      </SidebarProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Toggle Sidebar" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Cookie settings" }),
    );

    // The sheet blocks clicks outside itself, so the dialog must wait for it.
    expect(renew).not.toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(400);
    });

    expect(screen.queryByRole("dialog", { name: "Sidebar" })).toBeNull();
    expect(renew).toHaveBeenCalledOnce();
  });
});
