import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { useEffect } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import {
  listAllExecutionsQueryFn,
  resetSharedLiveUpdateLog,
  useLiveUpdateLog,
} from "../useLiveUpdateLog";

let capturedLog: ReturnType<typeof useLiveUpdateLog> | undefined;

function FetchesList({ enabled }: { enabled: boolean }) {
  const log = useLiveUpdateLog();
  useEffect(() => {
    capturedLog = log;
  }, [log]);
  const { data } = useGetV1ListAllExecutions({
    query: {
      queryFn: listAllExecutionsQueryFn,
      retry: 1,
      retryDelay: 0,
      enabled,
    },
  });
  return <div>{data ? "loaded" : "loading"}</div>;
}

describe("the shared run-list query function", () => {
  beforeEach(() => {
    resetSharedLiveUpdateLog();
    capturedLog = undefined;
  });

  test("it records each attempt's start, including a retry", async () => {
    let requests = 0;
    server.use(
      http.get("/api/proxy/api/executions", () => {
        requests += 1;
        if (requests === 1) {
          return HttpResponse.json({ detail: "boom" }, { status: 500 });
        }
        return HttpResponse.json([], { status: 200 });
      }),
    );
    const view = render(<FetchesList enabled={false} />);
    expect(capturedLog).toBeDefined();
    const recordStart = vi.spyOn(capturedLog!, "recordFetchStart");
    view.rerender(<FetchesList enabled />);

    expect(await screen.findByText("loaded")).toBeDefined();
    // The initial attempt and its retry each record a start: a retry
    // dispatches no "fetch" action, so only the query function can time it.
    expect(recordStart).toHaveBeenCalledTimes(2);
    recordStart.mockRestore();
  });
});
