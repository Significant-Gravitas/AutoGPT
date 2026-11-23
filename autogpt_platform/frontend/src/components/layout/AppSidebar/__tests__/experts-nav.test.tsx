import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertIdentity } from "@/app/api/__generated__/models/expertIdentity";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
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

const location = vi.hoisted(() => ({ pathname: "/team", search: "" }));

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
    usePathname: () => location.pathname,
    useSearchParams: () => new URLSearchParams(location.search),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

function expert(overrides: Partial<ExpertIdentity>): ExpertIdentity {
  return {
    id: "expert-maria",
    name: "Maria Ruiz",
    avatar_url: null,
    role: "Marketing",
    job_title: "Growth Lead",
    is_archived: false,
    ...overrides,
  };
}

const identitiesRequest = vi.fn(() => [
  expert({}),
  expert({ id: "expert-sam", name: "Sam Okafor", job_title: "Data Analyst" }),
  expert({ id: "expert-lee", name: "Lee Park", is_archived: true }),
]);

function renderSidebar({ open = true } = {}) {
  return render(
    <SidebarProvider defaultOpen={open}>
      <AppSidebar />
    </SidebarProvider>,
  );
}

function expertLinks() {
  const group = screen
    .getByRole("button", { name: "Experts" })
    .closest('[data-sidebar="group"]');
  return Array.from(group?.querySelectorAll("a[href]") ?? []);
}

beforeEach(() => {
  auth.isLoggedIn = true;
  location.pathname = "/team";
  location.search = "";
  identitiesRequest.mockClear();
  server.use(getListExpertIdentitiesMockHandler(identitiesRequest));
});

afterEach(() => {
  server.resetHandlers();
});

describe("Experts in the sidebar", () => {
  it("lists Otto first, then each hired expert, each starting a new chat", async () => {
    renderSidebar();

    expect(await screen.findByText("Sam Okafor")).toBeDefined();
    const links = expertLinks();
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/home",
      "/home?expertId=expert-maria&new=1",
      "/home?expertId=expert-sam&new=1",
    ]);
    expect(links[0].textContent).toContain("Otto");
    expect(links[1].textContent).toContain("Maria Ruiz");
    expect(links[1].textContent).toContain("Growth Lead");
  });

  it("leaves out experts who were let go", async () => {
    renderSidebar();

    expect(await screen.findByText("Maria Ruiz")).toBeDefined();
    expect(screen.queryByText("Lee Park")).toBeNull();
  });

  it("still offers Otto when no expert is hired", async () => {
    identitiesRequest.mockReturnValueOnce([]);
    renderSidebar();

    await waitFor(() => expect(identitiesRequest).toHaveBeenCalled());
    expect(expertLinks().map((link) => link.getAttribute("href"))).toEqual([
      "/home",
    ]);
  });

  it("marks the expert whose chat is open on Home as current", async () => {
    location.pathname = "/home";
    location.search = "expertId=expert-sam";
    renderSidebar();

    await screen.findByText("Sam Okafor");
    const current = expertLinks().filter(
      (link) => link.getAttribute("data-active") === "true",
    );
    expect(current.map((link) => link.textContent)).toEqual([
      expect.stringContaining("Sam Okafor"),
    ]);
  });

  it("is hidden in the collapsed icon rail", async () => {
    renderSidebar({ open: false });

    await screen.findByText("Maria Ruiz");
    // happy-dom applies no CSS: the wrapper's class is what hides it once the
    // desktop sidebar collapses to data-collapsible="icon".
    const wrapper = document.querySelector('[data-tour-target="chats"]');
    expect(
      wrapper?.classList.contains("group-data-[collapsible=icon]:hidden"),
    ).toBe(true);
    expect(wrapper?.closest('[data-collapsible="icon"]')).not.toBeNull();
  });

  it("is not shown to a logged-out visitor", () => {
    auth.isLoggedIn = false;
    renderSidebar();

    expect(screen.getByText("Marketplace")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Experts" })).toBeNull();
    expect(screen.queryByText("Otto")).toBeNull();
  });
});
