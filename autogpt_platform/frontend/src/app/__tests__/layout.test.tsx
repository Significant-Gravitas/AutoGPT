import Script from "next/script";
import { isValidElement, type ReactElement, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import RootLayout from "../layout";

vi.mock("@/components/styles/fonts", () => ({
  fonts: {
    poppins: { variable: "" },
    sans: { variable: "" },
    mono: { variable: "" },
  },
}));

// Only the layout's own markup is under test; the app shell it wraps pulls in
// Next.js internals that need the React canary Next.js vendors.
vi.mock("@/app/providers", () => ({
  Providers: ({ children }: { children: ReactNode }) => children,
}));
vi.mock("@/components/molecules/ErrorBoundary/ErrorBoundary", () => ({
  ErrorBoundary: ({ children }: { children: ReactNode }) => children,
}));
vi.mock("@/components/molecules/TallyPoup/TallyPopup", () => ({
  default: () => null,
}));
vi.mock("@/components/molecules/Toast/toaster", () => ({
  Toaster: () => null,
}));
vi.mock("@/services/analytics", () => ({ SetupAnalytics: () => null }));
vi.mock("@/services/analytics/VercelAnalyticsWrapper", () => ({
  VercelAnalyticsWrapper: () => null,
}));
vi.mock("@/components/AgentationDevtool", () => ({ default: () => null }));
vi.mock("@tanstack/react-query-devtools", () => ({
  ReactQueryDevtools: () => null,
}));

interface ScriptProps {
  id?: string;
  src?: string;
  strategy?: string;
  "data-cbid"?: string;
  "data-georegions"?: string;
  "data-blockingmode"?: string;
  "data-cookieconsent"?: string;
  dangerouslySetInnerHTML?: { __html: string };
}

// The layout is an async server component, so its element tree is inspected
// directly instead of being rendered into the DOM.
function findScripts(node: ReactNode): ScriptProps[] {
  if (Array.isArray(node)) return node.flatMap(findScripts);
  if (!isValidElement(node)) return [];
  const element = node as ReactElement<{ children?: ReactNode }>;
  if (element.type === Script) return [element.props as ScriptProps];
  return findScripts(element.props.children);
}

async function renderedScripts() {
  return findScripts(await RootLayout({ children: null }));
}

describe("RootLayout consent scripts", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("loads no banner and no Consent Mode defaults without a Cookiebot domain group", async () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");

    expect(await renderedScripts()).toEqual([]);
  });

  it("queues the Consent Mode defaults before hydration and loads Cookiebot after it", async () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "test-cbid");

    const [defaults, cookiebot, ...rest] = await renderedScripts();

    expect(rest).toEqual([]);
    expect(defaults).toMatchObject({
      id: "google-consent-defaults",
      strategy: "beforeInteractive",
      "data-cookieconsent": "ignore",
    });
    expect(defaults.dangerouslySetInnerHTML?.__html).toContain(
      "gtag('consent','default'",
    );
    expect(cookiebot).toEqual({
      id: "Cookiebot",
      src: "https://consent.cookiebot.com/uc.js",
      "data-cbid": "test-cbid",
      "data-blockingmode": "manual",
      strategy: "afterInteractive",
    });
    expect(cookiebot["data-georegions"]).toBeUndefined();
  });

  it("passes regional domain groups through verbatim", async () => {
    const geoRegions =
      "{'region':'US-06','cbid':'11111111-1111-1111-1111-111111111111'},{'region':'US','cbid':'22222222-2222-2222-2222-222222222222'}";
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "test-cbid");
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_GEOREGIONS", geoRegions);

    const cookiebot = (await renderedScripts()).find(
      (script) => script.id === "Cookiebot",
    );

    expect(cookiebot?.["data-georegions"]).toBe(geoRegions);
  });
});
