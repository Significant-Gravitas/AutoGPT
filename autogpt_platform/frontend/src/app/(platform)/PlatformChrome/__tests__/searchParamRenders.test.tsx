import { act, render, screen, waitFor } from "@testing-library/react";
import {
  AppRouterContext,
  type AppRouterInstance,
} from "next/dist/shared/lib/app-router-context.shared-runtime";
import {
  PathnameContext,
  SearchParamsContext,
} from "next/dist/shared/lib/hooks-client-context.shared-runtime";
import { useState, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SidebarProvider } from "@/components/ui/sidebar";

import { InsetHeaderSidebarTrigger } from "../components/InsetHeaderSidebarTrigger/InsetHeaderSidebarTrigger";
import { useIsChatSession } from "../useIsChatSession";
import { usePlatformChrome } from "../usePlatformChrome";

// The real hooks, reading the same contexts the App Router provides, so a
// search-param write reaches the components exactly as it does in the app.
vi.mock("next/navigation", async () => vi.importActual("next/navigation"));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => true };
});

interface URLState {
  pathname: string;
  searchParams: URLSearchParams;
}

const router = {
  push: vi.fn(),
  replace: vi.fn(),
  prefetch: vi.fn(),
  back: vi.fn(),
  forward: vi.fn(),
  refresh: vi.fn(),
} satisfies AppRouterInstance;

let navigate: (pathname: string, search: string) => void = () => {};

function FakeRouter({
  initialPathname,
  initialSearch,
  children,
}: {
  initialPathname: string;
  initialSearch: string;
  children: ReactNode;
}) {
  const [url, setURL] = useState<URLState>({
    pathname: initialPathname,
    searchParams: new URLSearchParams(initialSearch),
  });
  navigate = (pathname, search) =>
    setURL({ pathname, searchParams: new URLSearchParams(search) });
  return (
    <AppRouterContext.Provider value={router}>
      <PathnameContext.Provider value={url.pathname}>
        <SearchParamsContext.Provider value={url.searchParams}>
          {children}
        </SearchParamsContext.Provider>
      </PathnameContext.Provider>
    </AppRouterContext.Provider>
  );
}

let chromeRenders = 0;
function ChromeConsumer() {
  const { isCopilotRoute } = usePlatformChrome();
  chromeRenders += 1;
  return <span data-testid="copilot-route">{String(isCopilotRoute)}</span>;
}

function ChatSessionConsumer() {
  return <span data-testid="chat-session">{String(useIsChatSession())}</span>;
}

function navigateTo(pathname: string, search: string) {
  act(() => navigate(pathname, search));
}

describe("platform chrome and search-param writes", () => {
  beforeEach(() => {
    chromeRenders = 0;
  });

  it("does not re-render a usePlatformChrome consumer when only a search param changes", async () => {
    render(
      <FakeRouter
        initialPathname="/library/agents/agent-1"
        initialSearch="activeItem=run-1"
      >
        <ChromeConsumer />
      </FakeRouter>,
    );
    // Mount, then the post-mount layout swap.
    await waitFor(() => expect(chromeRenders).toBe(2));

    navigateTo("/library/agents/agent-1", "activeItem=run-2");
    navigateTo("/library/agents/agent-1", "activeItem=run-2&activeTab=runs");
    navigateTo("/copilot", "sessionId=chat-1");
    const afterPathnameChange = chromeRenders;
    navigateTo("/copilot", "sessionId=chat-2");

    expect(afterPathnameChange).toBe(3);
    expect(chromeRenders).toBe(3);
    expect(screen.getByTestId("copilot-route").textContent).toBe("true");
  });

  it("reports an open chat only on the copilot routes", () => {
    render(
      <FakeRouter initialPathname="/home" initialSearch="">
        <ChatSessionConsumer />
      </FakeRouter>,
    );
    const isChatSession = () => screen.getByTestId("chat-session").textContent;
    expect(isChatSession()).toBe("false");

    navigateTo("/home", "sessionId=chat-1");
    expect(isChatSession()).toBe("true");

    navigateTo("/copilot", "sessionId=chat-2");
    expect(isChatSession()).toBe("true");

    navigateTo("/copilot", "");
    expect(isChatSession()).toBe("false");

    navigateTo("/library", "sessionId=chat-1");
    expect(isChatSession()).toBe("false");
  });

  it("hides the floating sidebar trigger while a chat is open", () => {
    render(
      <FakeRouter initialPathname="/copilot" initialSearch="">
        <SidebarProvider>
          <InsetHeaderSidebarTrigger />
        </SidebarProvider>
      </FakeRouter>,
    );
    expect(screen.getByText("Toggle Sidebar")).toBeDefined();

    navigateTo("/copilot", "sessionId=chat-1");
    expect(screen.queryByText("Toggle Sidebar")).toBeNull();

    navigateTo("/copilot", "");
    expect(screen.getByText("Toggle Sidebar")).toBeDefined();
  });
});
