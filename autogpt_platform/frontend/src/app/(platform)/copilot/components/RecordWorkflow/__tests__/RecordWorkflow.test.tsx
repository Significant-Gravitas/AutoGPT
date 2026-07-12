import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";
import type { RecordingStartRequestInterpretationRoute } from "@/app/api/__generated__/models/recordingStartRequestInterpretationRoute";
import type { LocalPCExecutorStatus } from "../../../hooks/useLocalPCExecutor";
import { RecordWorkflow } from "../RecordWorkflow";

const EXECUTOR_URL =
  "http://localhost:3000/api/proxy/api/copilot/sessions/:sessionId/executor";
const START_URL = `${EXECUTOR_URL}/recording/start`;
const STOP_URL = `${EXECUTOR_URL}/recording/stop`;
const REVIEW_URL = `${EXECUTOR_URL}/recording/:recordingId/review`;

function useRecordingExecutor(overrides: Partial<LocalPCExecutorStatus> = {}) {
  server.use(
    http.get(EXECUTOR_URL, () =>
      HttpResponse.json({
        kind: "shim",
        platform: "darwin",
        capabilities: ["files", "shell", "recording"],
        computer_use_features: [],
        recording_routes: [
          "extract_then_cloud",
          "local_vlm",
          "screenshots_to_cloud",
        ],
        recording_channels: ["desktop_ax", "floor", "browser"],
        ...overrides,
      }),
    ),
  );
}

function recordingResponse(
  interpretationRoute: RecordingStartRequestInterpretationRoute = "extract_then_cloud",
) {
  return {
    summary: {
      recording_id: "rec-123",
      step_count: 2,
      enrichment_coverage: { dom: 1, ax: 0, none: 1 },
      duration_seconds: 12.5,
    },
    recording: {
      recording_id: "rec-123",
      version: "1.0",
      created_at: 123,
      machine_id: "machine-1",
      interpretation_route: interpretationRoute,
      redaction_applied: false,
      steps: [
        {
          seq: 1,
          action: "click",
          active_app: "Chrome",
          active_window: "Customer form",
          value: null,
        },
        {
          seq: 2,
          action: "type",
          active_app: "Chrome",
          active_window: "Email",
          value: { raw: "person@example.com" },
        },
      ],
    },
  };
}

describe("RecordWorkflow", () => {
  test("starts recording through the owner-scoped backend API", async () => {
    useRecordingExecutor();
    const received = vi.fn();
    server.use(
      http.post(START_URL, async ({ request }) => {
        received(await request.json());
        return HttpResponse.json({ recording_id: "rec-123" });
      }),
      http.post(STOP_URL, () => HttpResponse.json(recordingResponse())),
    );

    render(<RecordWorkflow sessionID="test-session-id" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );

    await waitFor(() =>
      expect(received).toHaveBeenCalledWith({
        mode: "demonstration",
        interpretation_route: "extract_then_cloud",
        channels: ["floor", "browser", "desktop_ax"],
      }),
    );
    expect(await screen.findByRole("button", { name: /stop/i })).toBeDefined();
    expect(screen.getByText(/^Recording$/)).toBeDefined();
  });

  test("reviews real captured steps and applies removals and redactions", async () => {
    useRecordingExecutor();
    const reviewRequest = vi.fn();
    server.use(
      http.post(START_URL, () =>
        HttpResponse.json({ recording_id: "rec-123" }),
      ),
      http.post(STOP_URL, () => HttpResponse.json(recordingResponse())),
      http.post(REVIEW_URL, async ({ request, params }) => {
        reviewRequest({ params, body: await request.json() });
        return HttpResponse.json({ recording_id: "rec-123", step_count: 1 });
      }),
    );

    render(<RecordWorkflow sessionID="test-session-id" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );
    fireEvent.click(await screen.findByRole("button", { name: /stop/i }));

    expect(
      (await screen.findAllByText("person@example.com")).length,
    ).toBeGreaterThan(0);
    fireEvent.click(screen.getByLabelText(/delete step 1/i));
    fireEvent.click(screen.getByLabelText(/hide value for step 2/i));
    fireEvent.click(screen.getByRole("button", { name: /finish review/i }));

    await waitFor(() =>
      expect(reviewRequest).toHaveBeenCalledWith({
        params: expect.objectContaining({ recordingId: "rec-123" }),
        body: {
          removed_step_seqs: [1],
          redacted_step_seqs: [2],
        },
      }),
    );
    expect(
      await screen.findByText(
        /recording reviewed. ask copilot to generate the skill/i,
      ),
    ).toBeDefined();
  });

  test("reports a start failure without pretending to record", async () => {
    useRecordingExecutor();
    server.use(
      http.post(START_URL, () =>
        HttpResponse.json({ detail: "consent declined" }, { status: 409 }),
      ),
    );

    render(<RecordWorkflow sessionID="test-session-id" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );

    expect((await screen.findByRole("alert")).textContent).toMatch(
      /could not start the recording/i,
    );
    expect(screen.queryByRole("button", { name: /stop/i })).toBeNull();
  });

  test("keeps the recording state when stop fails so the user can retry", async () => {
    useRecordingExecutor();
    let stopAttempts = 0;
    server.use(
      http.post(START_URL, () =>
        HttpResponse.json({ recording_id: "rec-123" }),
      ),
      http.post(STOP_URL, () => {
        stopAttempts += 1;
        return stopAttempts === 1
          ? HttpResponse.json({ detail: "shim timeout" }, { status: 504 })
          : HttpResponse.json(recordingResponse());
      }),
    );

    render(<RecordWorkflow sessionID="test-session-id" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );
    fireEvent.click(await screen.findByRole("button", { name: /stop/i }));

    expect((await screen.findByRole("alert")).textContent).toMatch(
      /could not stop the recording/i,
    );
    expect(screen.getByRole("button", { name: /stop/i })).toBeDefined();
  });

  test("requires fresh screenshot consent for every cloud recording", async () => {
    useRecordingExecutor({
      recording_routes: ["screenshots_to_cloud"],
      recording_channels: ["floor"],
    });
    const reviewRequest = vi.fn();
    server.use(
      http.post(START_URL, () =>
        HttpResponse.json({ recording_id: "rec-cloud" }),
      ),
      http.post(STOP_URL, () =>
        HttpResponse.json(recordingResponse("screenshots_to_cloud")),
      ),
      http.post(REVIEW_URL, async ({ request }) => {
        reviewRequest(await request.json());
        return HttpResponse.json({
          recording_id: "rec-cloud",
          step_count: 2,
        });
      }),
    );

    render(<RecordWorkflow sessionID="test-session-id" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );
    fireEvent.click(await screen.findByRole("button", { name: /stop/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /finish review/i }),
    );

    expect(
      await screen.findByText(/allow cloud processing for this recording/i),
    ).toBeDefined();
    expect(reviewRequest).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("button", { name: /allow cloud processing/i }),
    );
    await waitFor(() => expect(reviewRequest).toHaveBeenCalledOnce());

    fireEvent.click(
      await screen.findByRole("button", { name: /record another/i }),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );
    fireEvent.click(await screen.findByRole("button", { name: /stop/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /finish review/i }),
    );

    expect(
      await screen.findByText(/allow cloud processing for this recording/i),
    ).toBeDefined();
    expect(reviewRequest).toHaveBeenCalledOnce();

    fireEvent.click(
      screen.getByRole("button", { name: /allow cloud processing/i }),
    );
    await waitFor(() => expect(reviewRequest).toHaveBeenCalledTimes(2));
  });

  test("stops the old session recording when the active chat changes", async () => {
    useRecordingExecutor();
    const stopRequest = vi.fn();
    server.use(
      http.post(START_URL, () =>
        HttpResponse.json({ recording_id: "rec-old-session" }),
      ),
      http.post(STOP_URL, async ({ params, request }) => {
        stopRequest({
          sessionID: String(params.sessionId),
          body: await request.json(),
        });
        return HttpResponse.json(recordingResponse());
      }),
    );

    const view = render(<RecordWorkflow sessionID="old-session" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );
    expect(await screen.findByRole("button", { name: /stop/i })).toBeDefined();

    view.rerender(<RecordWorkflow sessionID="new-session" />);

    await waitFor(() =>
      expect(stopRequest).toHaveBeenCalledWith({
        sessionID: "old-session",
        body: { recording_id: "rec-old-session" },
      }),
    );
    expect(stopRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({ sessionID: "new-session" }),
    );
    expect(
      await screen.findByRole("button", { name: /record workflow/i }),
    ).toBeDefined();
  });

  test("stops a recording that starts after its UI unmounts", async () => {
    useRecordingExecutor();
    const startRequest = vi.fn();
    const stopRequest = vi.fn();
    let releaseStart = () => {};
    const startGate = new Promise<void>((resolve) => {
      releaseStart = resolve;
    });
    server.use(
      http.post(START_URL, async () => {
        startRequest();
        await startGate;
        return HttpResponse.json({ recording_id: "rec-after-unmount" });
      }),
      http.post(STOP_URL, async ({ params, request }) => {
        stopRequest({
          sessionID: String(params.sessionId),
          body: await request.json(),
        });
        return HttpResponse.json(recordingResponse());
      }),
    );

    const view = render(<RecordWorkflow sessionID="old-session" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );
    await waitFor(() => expect(startRequest).toHaveBeenCalledOnce());
    view.unmount();
    releaseStart();

    await waitFor(() =>
      expect(stopRequest).toHaveBeenCalledWith({
        sessionID: "old-session",
        body: { recording_id: "rec-after-unmount" },
      }),
    );
  });

  test("retries a failed in-flight stop after the recording UI unmounts", async () => {
    useRecordingExecutor();
    let stopAttempts = 0;
    let releaseFirstStop = () => {};
    const firstStopGate = new Promise<void>((resolve) => {
      releaseFirstStop = resolve;
    });
    server.use(
      http.post(START_URL, () =>
        HttpResponse.json({ recording_id: "rec-retry-stop" }),
      ),
      http.post(STOP_URL, async () => {
        stopAttempts += 1;
        if (stopAttempts === 1) {
          await firstStopGate;
          return HttpResponse.json(
            { detail: "temporary shim timeout" },
            { status: 504 },
          );
        }
        return HttpResponse.json(recordingResponse());
      }),
    );

    const view = render(<RecordWorkflow sessionID="old-session" />);
    fireEvent.click(
      await screen.findByRole("button", { name: /record workflow/i }),
    );
    fireEvent.click(await screen.findByRole("button", { name: /stop/i }));
    await waitFor(() => expect(stopAttempts).toBe(1));

    view.unmount();
    releaseFirstStop();

    await waitFor(() => expect(stopAttempts).toBe(2));
  });

  test("does not start recording without advertised routes and channels", async () => {
    const requested = vi.fn();
    server.use(
      http.get(EXECUTOR_URL, () => {
        requested();
        return HttpResponse.json({
          kind: "shim",
          capabilities: ["recording"],
          recording_routes: [],
          recording_channels: [],
        });
      }),
    );

    render(<RecordWorkflow sessionID="test-session-id" />);

    await waitFor(() => expect(requested).toHaveBeenCalledOnce());
    expect(
      screen.queryByRole("button", { name: /record workflow/i }),
    ).toBeNull();
  });

  test("does not render when the shim lacks recording capability", async () => {
    const requested = vi.fn();
    server.use(
      http.get(EXECUTOR_URL, () => {
        requested();
        return HttpResponse.json({
          kind: "shim",
          capabilities: ["files", "shell"],
        });
      }),
    );

    render(<RecordWorkflow sessionID="test-session-id" />);

    await waitFor(() => expect(requested).toHaveBeenCalledOnce());
    expect(
      screen.queryByRole("button", { name: /record workflow/i }),
    ).toBeNull();
  });
});
