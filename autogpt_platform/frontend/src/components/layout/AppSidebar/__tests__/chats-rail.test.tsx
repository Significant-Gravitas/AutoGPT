import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";

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

const viewport = vi.hoisted(() => ({ isMobile: false }));

vi.mock("@/hooks/use-mobile", () => ({
  useIsMobile: () => viewport.isMobile,
}));

// happy-dom applies no CSS and has no Web Animations, so the tests stub the
// scroll and drive the "animations finished" signal themselves.
const scrollTo = vi.fn();
const getAnimations = vi.fn<() => Animation[]>(() => []);
const originalScrollTo = Element.prototype.scrollTo;

function runningAnimation() {
  let finish!: () => void;
  const animation = {
    playState: "running",
    effect: { getComputedTiming: () => ({ endTime: 260 }) },
  } as unknown as Animation;
  Object.defineProperty(animation, "finished", {
    value: new Promise<Animation>((resolve) => {
      finish = () => {
        Object.assign(animation, { playState: "finished" });
        resolve(animation);
      };
    }),
  });
  return { animation, finish };
}

function renderSidebar({ open = false } = {}) {
  return render(
    <SidebarProvider defaultOpen={open}>
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

function toggleSidebarWithShortcut() {
  fireEvent.keyDown(window, { key: "b", ctrlKey: true });
}

beforeEach(() => {
  auth.isLoggedIn = true;
  motionPreference.reduced = false;
  viewport.isMobile = false;
  useGetFlagMock.mockReturnValue(true);
  scrollTo.mockReset();
  getAnimations.mockReset();
  getAnimations.mockReturnValue([]);
  Element.prototype.scrollTo = scrollTo;
  Element.prototype.getAnimations = getAnimations;
  server.use(getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }));
});

afterEach(() => {
  Element.prototype.scrollTo = originalScrollTo;
  Reflect.deleteProperty(Element.prototype, "getAnimations");
  vi.unstubAllGlobals();
});

describe("Chats in the collapsed sidebar", () => {
  it("shows a Chats button in the collapsed rail for a logged-in user", () => {
    renderSidebar();

    expect(getSidebarState()).toBe("collapsed");
    const chats = getChatsButton();
    expect(chats.tagName).toBe("BUTTON");
    expect(chats.getAttribute("data-active")).toBe("false");
  });

  it("is only displayed inside the collapsed icon rail", () => {
    renderSidebar({ open: true });

    // `hidden` keeps it out of the expanded sidebar and the phone sheet; only
    // the desktop sidebar's data-collapsible="icon" group displays it.
    const wrapper = getChatsButton().closest(
      '[data-sidebar="group"]',
    )?.parentElement;
    expect(wrapper?.classList.contains("hidden")).toBe(true);
    expect(
      wrapper?.classList.contains("group-data-[collapsible=icon]:block"),
    ).toBe(true);
  });

  it("expands the sidebar, then scrolls only the sidebar to Recent chats and focuses its heading", async () => {
    const user = userEvent.setup();
    renderSidebar();
    const statesAtScroll: unknown[] = [];
    scrollTo.mockImplementation(() => statesAtScroll.push(getSidebarState()));
    const statesAtFocus: unknown[] = [];
    getRecentChatsHeading().addEventListener("focus", () =>
      statesAtFocus.push(getSidebarState()),
    );
    const focusHeading = vi.spyOn(getRecentChatsHeading(), "focus");

    await user.click(getChatsButton());

    expect(getSidebarState()).toBe("expanded");
    await waitFor(() => expect(statesAtScroll).toEqual(["expanded"]));
    expect(statesAtFocus).toEqual(["expanded"]);
    expect(focusHeading).toHaveBeenCalledWith({ preventScroll: true });
    expect(scrollTo.mock.contexts[0]).toBe(getSidebarScrollArea());
    expect(scrollTo).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: "smooth" }),
    );
    expect(document.activeElement).toBe(getRecentChatsHeading());
  });

  it("jumps from the keyboard", async () => {
    const user = userEvent.setup();
    renderSidebar();

    getChatsButton().focus();
    await user.keyboard("{Enter}");

    expect(getSidebarState()).toBe("expanded");
    await waitFor(() => expect(scrollTo).toHaveBeenCalledTimes(1));
    expect(document.activeElement).toBe(getRecentChatsHeading());
  });

  it("scrolls without animation when the user prefers reduced motion", async () => {
    motionPreference.reduced = true;
    const user = userEvent.setup();
    renderSidebar();

    await user.click(getChatsButton());

    await waitFor(() => expect(scrollTo).toHaveBeenCalledTimes(1));
    expect(scrollTo).toHaveBeenCalledWith(
      expect.objectContaining({ behavior: "auto" }),
    );
  });

  it("reopens Recent chats when the user had closed it, and scrolls once the sidebar has finished animating", async () => {
    const user = userEvent.setup();
    renderSidebar();
    expect(await screen.findByText("No conversations yet")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Expand sidebar" }));
    await user.click(getRecentChatsHeading());
    await waitFor(() =>
      expect(screen.queryByText("No conversations yet")).toBeNull(),
    );
    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    expect(getSidebarState()).toBe("collapsed");
    const opening = runningAnimation();
    getAnimations.mockReturnValue([opening.animation]);
    let headingTop = 400;
    vi.spyOn(
      getRecentChatsHeading(),
      "getBoundingClientRect",
    ).mockImplementation(() => new DOMRect(0, headingTop, 0, 0));

    await user.click(getChatsButton());

    expect(getSidebarState()).toBe("expanded");
    expect(await screen.findByText("No conversations yet")).toBeDefined();
    expect(getRecentChatsHeading().getAttribute("aria-expanded")).toBe("true");
    expect(document.activeElement).toBe(getRecentChatsHeading());
    expect(scrollTo).not.toHaveBeenCalled();

    await act(async () => {
      headingTop = 420;
      opening.finish();
    });

    await waitFor(() => expect(scrollTo).toHaveBeenCalledTimes(1));
    expect(scrollTo.mock.contexts[0]).toBe(getSidebarScrollArea());
    expect(scrollTo).toHaveBeenCalledWith(
      expect.objectContaining({ top: 420 }),
    );
    expect(getAnimations.mock.contexts).not.toHaveLength(0);
    for (const element of getAnimations.mock.contexts) {
      expect(element).toBe(getSidebarScrollArea());
    }
  });

  it("keeps scrolling while the Recent chats list grows, until the sidebar collapses", async () => {
    const observers: {
      callback: () => void;
      observe: Mock;
      disconnect: Mock;
    }[] = [];
    vi.stubGlobal(
      "ResizeObserver",
      class {
        callback: () => void;
        observe = vi.fn();
        unobserve = vi.fn();
        disconnect = vi.fn();
        constructor(callback: () => void) {
          this.callback = callback;
          observers.push(this);
        }
      },
    );
    const user = userEvent.setup();
    renderSidebar();
    expect(await screen.findByText("No conversations yet")).toBeDefined();
    vi.spyOn(getRecentChatsHeading(), "getBoundingClientRect").mockReturnValue(
      new DOMRect(0, 420, 0, 0),
    );

    await user.click(getChatsButton());
    await waitFor(() => expect(scrollTo).toHaveBeenCalledTimes(1));

    const followers = observers.filter((observer) =>
      observer.observe.mock.calls.some(([element]) =>
        element.contains(getRecentChatsHeading()),
      ),
    );
    expect(followers).toHaveLength(1);
    const [watched] = followers[0].observe.mock.calls[0];
    expect(watched).not.toBe(getSidebarScrollArea());
    expect(watched.contains(screen.getByText("No conversations yet"))).toBe(
      true,
    );
    followers[0].callback();
    expect(scrollTo).toHaveBeenCalledTimes(2);
    expect(followers[0].disconnect).not.toHaveBeenCalled();

    toggleSidebarWithShortcut();

    expect(followers[0].disconnect).toHaveBeenCalled();
  });

  it("does not scroll if the sidebar collapses again before it finishes animating", async () => {
    const user = userEvent.setup();
    renderSidebar();
    const expanding = runningAnimation();
    getAnimations.mockReturnValue([expanding.animation]);

    await user.click(getChatsButton());
    expect(getSidebarState()).toBe("expanded");
    toggleSidebarWithShortcut();
    expect(getSidebarState()).toBe("collapsed");
    await act(async () => expanding.finish());

    expect(scrollTo).not.toHaveBeenCalled();
  });

  it("does not jump again when the sidebar is later expanded another way", async () => {
    const user = userEvent.setup();
    renderSidebar();

    await user.click(getChatsButton());
    await waitFor(() => expect(scrollTo).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    await user.click(screen.getByRole("button", { name: "Expand sidebar" }));

    expect(getSidebarState()).toBe("expanded");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(scrollTo).toHaveBeenCalledTimes(1);
  });

  it("does not show Chats to a logged-out visitor", () => {
    auth.isLoggedIn = false;
    renderSidebar();

    expect(screen.queryByRole("button", { name: "Chats" })).toBeNull();
  });

  it("still reopens Recent chats each time the phone sheet opens", async () => {
    viewport.isMobile = true;
    const user = userEvent.setup();
    renderSidebar();

    toggleSidebarWithShortcut();
    expect(await screen.findByText("No conversations yet")).toBeDefined();
    await user.click(getRecentChatsHeading());
    await waitFor(() =>
      expect(screen.queryByText("No conversations yet")).toBeNull(),
    );

    toggleSidebarWithShortcut();
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Recent chats" })).toBeNull(),
    );
    toggleSidebarWithShortcut();

    expect(await screen.findByText("No conversations yet")).toBeDefined();
    expect(getRecentChatsHeading().getAttribute("aria-expanded")).toBe("true");
  });
});
