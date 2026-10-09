import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppSidebar } from "../AppSidebar";

const auth = vi.hoisted(() => ({ isLoggedIn: true }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: auth.isLoggedIn
      ? { id: "user-1", email: "alice@example.com", role: "user" }
      : null,
    isLoggedIn: auth.isLoggedIn,
    isUserLoading: false,
  }),
}));

// The global next/link mock only exports `default`; AppSidebar also imports
// `useLinkStatus`, so re-mock here with a no-op pending status.
vi.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
    ...props
  }: {
    children: ReactNode;
    href: string;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
    usePathname: () => "/team",
    useSearchParams: () => new URLSearchParams(),
  };
});

const useGetFlagMock = vi.hoisted(() =>
  vi.fn<(flag: Flag) => boolean>(() => true),
);

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: Flag) => useGetFlagMock(flag),
  };
});

const motionPreference = vi.hoisted(() => ({ reduced: false }));

vi.mock("framer-motion", async (importOriginal) => {
  const actual = await importOriginal<typeof import("framer-motion")>();
  return {
    ...actual,
    useReducedMotion: () => motionPreference.reduced,
  };
});

const scrollTo = vi.fn();
const originalScrollTo = Element.prototype.scrollTo;

function renderCollapsedSidebar() {
  return render(
    <SidebarProvider defaultOpen={false}>
      <AppSidebar />
    </SidebarProvider>,
  );
}

function getSidebarState() {
  return document
    .querySelector('[data-sidebar="sidebar"]')
    ?.closest("[data-state]")
    ?.getAttribute("data-state");
}

function getSidebarScrollArea() {
  return document.querySelector('[data-sidebar="content"]');
}

function getChatsButton() {
  return screen.getByRole("button", { name: "Chats" });
}

function getRecentChatsHeading() {
  return screen.getByRole("button", { name: "Recent chats" });
}

beforeEach(() => {
  auth.isLoggedIn = true;
  motionPreference.reduced = false;
  useGetFlagMock.mockReturnValue(true);
  scrollTo.mockReset();
  Element.prototype.scrollTo = scrollTo;
  server.use(getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }));
});

afterEach(() => {
  Element.prototype.scrollTo = originalScrollTo;
});

describe("Chats in the collapsed sidebar", () => {
  it("shows a Chats button in the collapsed rail for a logged-in user", () => {
    renderCollapsedSidebar();

    expect(getSidebarState()).toBe("collapsed");
    const chats = getChatsButton();
    expect(chats.tagName).toBe("BUTTON");
    expect(chats.getAttribute("data-active")).toBe("false");
  });

  it("expands the sidebar, scrolls only the sidebar to Recent chats and focuses its heading", async () => {
    const user = userEvent.setup();
    renderCollapsedSidebar();

    await user.click(getChatsButton());

    expect(getSidebarState()).toBe("expanded");
    expect(scrollTo).toHaveBeenCalledTimes(1);
    expect(scrollTo.mock.contexts[0]).toBe(getSidebarScrollArea());
    expect(scrollTo).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: "smooth" }),
    );
    expect(document.activeElement).toBe(getRecentChatsHeading());
  });

  it("jumps from the keyboard", async () => {
    const user = userEvent.setup();
    renderCollapsedSidebar();

    getChatsButton().focus();
    await user.keyboard("{Enter}");

    expect(getSidebarState()).toBe("expanded");
    expect(scrollTo).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(getRecentChatsHeading());
  });

  it("scrolls without animation when the user prefers reduced motion", async () => {
    motionPreference.reduced = true;
    const user = userEvent.setup();
    renderCollapsedSidebar();

    await user.click(getChatsButton());

    expect(scrollTo).toHaveBeenCalledTimes(1);
    expect(scrollTo).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: "auto" }),
    );
  });

  it("reopens Recent chats when the user had closed it, then scrolls once it has opened", async () => {
    const user = userEvent.setup();
    renderCollapsedSidebar();
    expect(await screen.findByText("No conversations yet")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Expand sidebar" }));
    await user.click(getRecentChatsHeading());
    await waitFor(() =>
      expect(screen.queryByText("No conversations yet")).toBeNull(),
    );
    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    expect(getSidebarState()).toBe("collapsed");

    fireEvent.click(getChatsButton());

    expect(getSidebarState()).toBe("expanded");
    expect(scrollTo).not.toHaveBeenCalled();
    expect(await screen.findByText("No conversations yet")).toBeDefined();
    expect(document.activeElement).toBe(getRecentChatsHeading());
    await waitFor(() => expect(scrollTo).toHaveBeenCalledTimes(1));
    expect(scrollTo.mock.contexts[0]).toBe(getSidebarScrollArea());
  });

  it("does not jump again when the sidebar is later expanded another way", async () => {
    const user = userEvent.setup();
    renderCollapsedSidebar();

    await user.click(getChatsButton());
    expect(scrollTo).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    await user.click(screen.getByRole("button", { name: "Expand sidebar" }));

    expect(getSidebarState()).toBe("expanded");
    expect(scrollTo).toHaveBeenCalledTimes(1);
  });

  it("does not show Chats to a logged-out visitor", () => {
    auth.isLoggedIn = false;
    renderCollapsedSidebar();

    expect(screen.queryByRole("button", { name: "Chats" })).toBeNull();
  });
});
