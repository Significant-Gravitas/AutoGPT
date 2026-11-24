import type { AnchorHTMLAttributes, ReactNode } from "react";
import { renderToString } from "react-dom/server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { isDrawnTransparent } from "@/tests/integrations/transparency";
import { afterEach, describe, expect, test, vi } from "vitest";
import SettingsLayout from "../layout";

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/settings/profile",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: function MockLink({
    children,
    href,
    ...props
  }: AnchorHTMLAttributes<HTMLAnchorElement> & {
    children: ReactNode;
    href: string;
  }) {
    return (
      <a href={href} {...props}>
        {children}
      </a>
    );
  },
  useLinkStatus: () => ({ pending: false }),
}));

function settingsPage() {
  return (
    <SettingsLayout>
      <h1>Profile settings</h1>
    </SettingsLayout>
  );
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("/settings first paint", () => {
  test("the server's HTML draws the section links and the page, before the JavaScript has run", () => {
    const container = document.createElement("div");
    container.innerHTML = renderToString(settingsPage());
    document.body.appendChild(container);

    const page = within(container).getByRole("heading", {
      name: "Profile settings",
    });
    const sectionLink = within(container).getByRole("link", {
      name: "Account",
    });
    expect(isDrawnTransparent(page)).toBe(false);
    expect(isDrawnTransparent(sectionLink)).toBe(false);
  });

  test("mounted after the page has loaded, the section links and the page still enter", () => {
    render(settingsPage());

    expect(
      isDrawnTransparent(
        screen.getByRole("heading", { name: "Profile settings" }),
      ),
    ).toBe(true);
    expect(
      isDrawnTransparent(screen.getByRole("link", { name: "Account" })),
    ).toBe(true);
  });
});
