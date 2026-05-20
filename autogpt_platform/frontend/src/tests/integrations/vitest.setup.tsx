import { beforeAll, afterAll, afterEach, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { mockNextjsModules } from "./setup-nextjs-mocks";
import { mockSupabaseRequest } from "./mock-supabase-request";
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

beforeAll(() => {
  mockNextjsModules();
  mockSupabaseRequest(); // If you need user's data - please mock supabase actions in your specific test - it sends null user [It's only to avoid cookies() call]
  return server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => {
  cleanup();
  server.resetHandlers();
});
afterAll(() => server.close());
