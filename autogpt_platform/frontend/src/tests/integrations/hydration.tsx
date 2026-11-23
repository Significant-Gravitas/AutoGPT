import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { useState, type ReactElement, type ReactNode } from "react";
import { renderToString } from "react-dom/server";
import { vi } from "vitest";

interface Args {
  ui: ReactElement;
  serverNow: Date;
  browserNow: Date;
}

/**
 * A page load: `ui` rendered to HTML while the clock reads `serverNow`, that
 * HTML in the document, then hydrated while the clock reads `browserNow`, as
 * for a visitor whose clock is off the server's. Needs a fake Date
 * (`vi.useFakeTimers({ toFake: ["Date"] })`).
 *
 * `hydrationErrors` holds what React recovered from by rendering the page
 * again in the browser; a text mismatch there is error #418 in production.
 */
export function hydrateAcrossClocks({ ui, serverNow, browserNow }: Args) {
  vi.setSystemTime(serverNow);
  const container = document.createElement("div");
  container.innerHTML = renderToString(<PageProviders>{ui}</PageProviders>);
  const serverText = container.textContent ?? "";
  document.body.appendChild(container);

  vi.setSystemTime(browserNow);
  const hydrationErrors: string[] = [];
  const result = render(<PageProviders>{ui}</PageProviders>, {
    container,
    hydrate: true,
    onRecoverableError: (error) => hydrationErrors.push(String(error)),
  });
  return { ...result, serverText, hydrationErrors };
}

// The server and the browser each get their own query client, as in the app.
// Nothing goes stale, so the browser keeps what the server sent even with its
// clock days ahead. Unlike the app's client, it never refetches or retries, so
// text that changes because of a refetch right after hydration won't show here.
function PageProviders({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: false, staleTime: Infinity } },
      }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter>
        <BackendAPIProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </BackendAPIProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>
  );
}
