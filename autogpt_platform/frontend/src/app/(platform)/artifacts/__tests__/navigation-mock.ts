import { useSyncExternalStore } from "react";
import { vi } from "vitest";

// The artifacts page keeps its open folder in the URL, so a static
// `useSearchParams` mock would leave the page frozen at root whatever the
// router is told. This store replays a router call as the query string the
// next render reads, which is what the browser does.
let searchParams = new URLSearchParams();
const listeners = new Set<() => void>();

export const routerMock = {
  push: vi.fn((href: string) => navigate(href)),
  replace: vi.fn((href: string) => navigate(href)),
  prefetch: vi.fn(),
  back: vi.fn(),
  forward: vi.fn(),
  refresh: vi.fn(),
};

export function navigationMock(opts: { onNotFound?: () => void } = {}) {
  return {
    useRouter: () => routerMock,
    usePathname: () => "/artifacts",
    useSearchParams: () =>
      useSyncExternalStore(subscribe, getSearchParams, getSearchParams),
    useParams: () => ({}),
    notFound: () => {
      opts.onNotFound?.();
      throw new Error("NEXT_NOT_FOUND");
    },
  };
}

// Call from `beforeEach` — `query` is the URL the page is entered on.
export function resetNavigation(query = "") {
  searchParams = new URLSearchParams(query);
  routerMock.push.mockClear();
  routerMock.replace.mockClear();
  listeners.forEach((notify) => notify());
}

export function currentFolderParam(): string | null {
  return searchParams.get("folder");
}

function navigate(href: string) {
  searchParams = new URLSearchParams(href.split("?")[1] ?? "");
  listeners.forEach((notify) => notify());
}

function subscribe(notify: () => void) {
  listeners.add(notify);
  return () => listeners.delete(notify);
}

function getSearchParams() {
  return searchParams;
}
