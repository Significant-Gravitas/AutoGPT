import { syncReplayWithPage } from "@/lib/session-replay";
import { render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ReplayRouteGuard } from "../ReplayRouteGuard";

const navigation = vi.hoisted(() => ({ pathname: "/link/t-1" }));

vi.mock("next/navigation", () => ({
  usePathname: () => navigation.pathname,
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/session-replay", () => ({ syncReplayWithPage: vi.fn() }));

afterEach(() => {
  history.replaceState({}, "", "/");
});

it("checks session replay against each page a navigation lands on", () => {
  history.replaceState({}, "", "/link/t-1");
  const { rerender } = render(<ReplayRouteGuard />);
  expect(syncReplayWithPage).toHaveBeenLastCalledWith(
    `${location.origin}/link/t-1`,
  );

  history.replaceState({}, "", "/library");
  navigation.pathname = "/library";
  rerender(<ReplayRouteGuard />);

  expect(syncReplayWithPage).toHaveBeenLastCalledWith(
    `${location.origin}/library`,
  );
  expect(syncReplayWithPage).toHaveBeenCalledTimes(2);
});
