import { beforeAll, afterAll, afterEach, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { mockNextjsModules } from "./setup-nextjs-mocks";
import { mockAuthRequest } from "./mock-auth-request";
import { cleanup } from "@testing-library/react";

// React 18.3 only ships `cache` under the `react-server` export condition.
// Vitest doesn't set that condition, so any module that imports `cache` from
// "react" (e.g. server-only helpers wrapped for per-request memoization) blows
// up with "cache is not a function" the moment its file is evaluated. Shim it
// to identity here — the cache contract degrades to "no deduplication," which
// is the correct semantics in a unit-test context.
vi.mock("react", async (importActual) => {
  const actual = await importActual<typeof import("react")>();
  return {
    ...actual,
    cache: <T extends (...args: unknown[]) => unknown>(fn: T): T => fn,
  };
});

// NumberFlow renders into a shadow-DOM custom element and schedules animation
// work outside React. happy-dom can mount it, but the resulting rAF + custom
// element lifecycle collides with React 18's concurrent renderer during
// cleanup ("Should not already be working"). Render a plain span with the
// formatted value — tests query the wrapping aria-label, not NumberFlow's
// internals.
vi.mock("@number-flow/react", () => ({
  default: ({
    value,
    format,
    locales,
  }: {
    value: number;
    format?: Intl.NumberFormatOptions;
    locales?: Intl.LocalesArgument;
  }) => {
    const text = format
      ? new Intl.NumberFormat(locales ?? "en-US", format).format(value)
      : String(value);
    return <span>{text}</span>;
  },
}));

// happy-dom 20.14 rejects `Animation.finished` in cancel() without marking it
// handled, so every framer-motion animation cancelled during a test surfaces
// as an "Unhandled Rejection" AbortError. The Web Animations spec sets
// [[PromiseIsHandled]] on that promise when cancelling, so browsers never
// report it: https://drafts.csswg.org/web-animations-1/#cancel-an-animation
if (typeof Animation !== "undefined") {
  const cancel = Animation.prototype.cancel;
  Animation.prototype.cancel = function cancelWithHandledFinished(
    this: Animation,
  ) {
    this.finished.catch(() => {});
    cancel.call(this);
  };
}

beforeAll(() => {
  mockNextjsModules();
  mockAuthRequest(); // If you need user's data - please mock auth actions in your specific test - it sends null user [It's only to avoid cookies() call]
  return server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => {
  cleanup();
  server.resetHandlers();
});
afterAll(() => server.close());
