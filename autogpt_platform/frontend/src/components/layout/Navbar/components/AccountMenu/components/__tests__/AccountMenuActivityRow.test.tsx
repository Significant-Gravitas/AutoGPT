import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import {
  AFTER_RECONNECT,
  installLiveSocket,
} from "@/tests/integrations/live-socket";
import { screen, within } from "@/tests/integrations/test-utils";
import { QueryClient } from "@tanstack/react-query";
import { render as renderWithoutProviders } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, expect, test, vi } from "vitest";
import { AccountMenuActivityRow } from "../AccountMenuActivityRow";
import {
  findRunLine,
  openActivity,
  providersFor,
  run,
  serveActivity,
} from "./account-menu-activity-harness";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

let releaseHeldList: (() => void) | undefined;

afterEach(() => {
  // Settle a held response even when the test failed before releasing it.
  releaseHeldList?.();
  releaseHeldList = undefined;
  vi.unstubAllGlobals();
});

test("a run that finished while the live connection was down shows as finished once it reconnects", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveActivity(async () => [run("run-1", status)]);
  const live = installLiveSocket();

  const list = await openActivity();
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();

  // The list keeps its length: only the run's status changes.
  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();

  expect(await findRunLine(list, /^Completed /, AFTER_RECONNECT)).toBeDefined();
  expect(within(list).queryByText(/running$/)).toBeNull();
});

test("a run's live completion survives a refetch that read the run before it finished", async () => {
  let releaseStaleList = () => {};
  const staleListHeld = new Promise<void>((resolve) => {
    releaseStaleList = resolve;
  });
  releaseHeldList = releaseStaleList;
  let requests = 0;
  serveActivity(async () => {
    requests += 1;
    if (requests === 1) return [run("run-1", AgentExecutionStatus.RUNNING)];
    // Read while run-1 was still going, answered after it finished.
    await staleListHeld;
    return [
      run("run-1", AgentExecutionStatus.RUNNING),
      run("run-2", AgentExecutionStatus.RUNNING),
    ];
  });
  const live = installLiveSocket();

  const list = await openActivity();
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();
  await live.drop(first);
  const second = await live.acceptNext();
  await vi.waitFor(() => expect(requests).toBe(2), AFTER_RECONNECT);

  await live.publish(second, {
    ...run("run-1", AgentExecutionStatus.COMPLETED),
    ended_at: new Date().toISOString(),
  });
  expect(await findRunLine(list, /^Completed /)).toBeDefined();
  releaseStaleList();

  // run-2 shows once the stale list has landed.
  expect(await within(list).findAllByText(/running$/)).toHaveLength(1);
  expect(within(list).getByText(/^Completed /)).toBeDefined();
});

test("a refetch that started after a live update shows what it read", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveActivity(async () => [run("run-1", status)]);
  const live = installLiveSocket();

  const list = await openActivity();
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();

  await live.publish(first, {
    ...run("run-1", AgentExecutionStatus.TERMINATED),
    ended_at: new Date().toISOString(),
  });
  expect(await findRunLine(list, /^Stopped /)).toBeDefined();

  // A stopped run can be resumed; a list read after the update is newer.
  status = AgentExecutionStatus.RUNNING;
  await live.drop(first);
  await live.acceptNext();

  expect(await findRunLine(list, /running$/, AFTER_RECONNECT)).toBeDefined();
  expect(within(list).queryByText(/^Stopped /)).toBeNull();
});

test("opening the menu again shows a run that finished while it was closed", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveActivity(async () => [run("run-1", status)]);
  installLiveSocket();
  // The app keeps a fetched list fresh for a minute
  // (lib/react-query/queryClient.ts).
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
  });
  // The row mounts only while the account menu is open.
  function AccountMenu({ open }: { open: boolean }) {
    return open ? <AccountMenuActivityRow /> : null;
  }

  const { rerender } = renderWithoutProviders(<AccountMenu open />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  let list = await screen.findByTestId("agent-activity-dropdown");
  expect(await findRunLine(list, /running$/)).toBeDefined();

  rerender(<AccountMenu open={false} />);
  status = AgentExecutionStatus.COMPLETED;
  rerender(<AccountMenu open />);
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  list = await screen.findByTestId("agent-activity-dropdown");

  expect(await findRunLine(list, /^Completed /)).toBeDefined();
});

test("a retried refetch isn't overruled by an update from before its first attempt failed", async () => {
  let releaseFailure = () => {};
  const failureHeld = new Promise<void>((resolve) => {
    releaseFailure = resolve;
  });
  releaseHeldList = releaseFailure;
  serveActivity(async () => [run("run-1", AgentExecutionStatus.RUNNING)]);
  let requests = 0;
  server.use(
    http.get("/api/proxy/api/executions", async () => {
      requests += 1;
      if (requests !== 2) {
        // Stopped, then resumed before the retry read it.
        return HttpResponse.json([run("run-1", AgentExecutionStatus.RUNNING)]);
      }
      await failureHeld;
      return HttpResponse.json({ detail: "Unavailable" }, { status: 503 });
    }),
  );
  const live = installLiveSocket();
  // The app retries a failed read (lib/react-query/queryClient.ts).
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: 1, retryDelay: 0 } },
  });
  renderWithoutProviders(<AccountMenuActivityRow />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  const list = await screen.findByTestId("agent-activity-dropdown");
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();

  await live.drop(first);
  const second = await live.acceptNext();
  await vi.waitFor(() => expect(requests).toBe(2), AFTER_RECONNECT);
  // The stop arrives while the reconnect read is out, and that read fails.
  await live.publish(second, {
    ...run("run-1", AgentExecutionStatus.TERMINATED),
    ended_at: new Date().toISOString(),
  });
  expect(await findRunLine(list, /^Stopped /)).toBeDefined();
  releaseFailure();

  await vi.waitFor(() => expect(requests).toBe(3));
  expect(await findRunLine(list, /running$/)).toBeDefined();
  expect(within(list).queryByText(/^Stopped /)).toBeNull();
});
