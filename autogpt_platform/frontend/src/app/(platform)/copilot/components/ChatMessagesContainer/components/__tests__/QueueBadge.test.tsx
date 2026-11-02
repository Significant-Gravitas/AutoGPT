import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { render } from "@/tests/integrations/test-utils";
import { QueueBadge } from "../QueueBadge";

const cancelMock = vi.fn();
const captureExceptionMock = vi.fn();
const toastMock = vi.fn();
let isPending = false;
type ErrorResponse = { response?: { status?: number } } | null;
let errorResponse: ErrorResponse = null;

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  usePostV2CancelSessionTask: ({
    mutation,
  }: {
    mutation?: {
      onSuccess?: (response: { status: number }) => void;
      onError?: (error: unknown) => void;
    };
  }) => ({
    mutate: (args: { sessionId: string }) => {
      cancelMock(args);
      if (errorResponse !== null) {
        mutation?.onError?.(errorResponse);
        return;
      }
      mutation?.onSuccess?.({ status: 200 });
    },
    isPending,
  }),
  getGetV2GetSessionQueryKey: (sessionId: string) => [
    `/api/chat/sessions/${sessionId}`,
  ],
}));

vi.mock("@sentry/nextjs", () => ({
  captureException: (...args: unknown[]) => captureExceptionMock(...args),
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastMock(...args),
  useToast: () => ({ toast: toastMock, dismiss: vi.fn(), toasts: [] }),
}));

beforeEach(() => {
  errorResponse = null;
});

afterEach(() => {
  cancelMock.mockClear();
  captureExceptionMock.mockClear();
  toastMock.mockClear();
  isPending = false;
});

describe("QueueBadge", () => {
  it("renders the queued badge with the cancel button when a sessionID is present", () => {
    render(<QueueBadge sessionID="sess-1" />);
    expect(screen.getByTestId("queue-badge-queued")).toBeDefined();
    expect(screen.getByTestId("queue-cancel-button")).toBeDefined();
  });

  it("invokes the cancel mutation with the session id on click", () => {
    render(<QueueBadge sessionID="sess-42" />);
    fireEvent.click(screen.getByTestId("queue-cancel-button"));
    expect(cancelMock).toHaveBeenCalledWith({ sessionId: "sess-42" });
  });

  it("hides the cancel button when no sessionID is available", () => {
    render(<QueueBadge sessionID={null} />);
    expect(screen.getByTestId("queue-badge-queued")).toBeDefined();
    expect(screen.queryByTestId("queue-cancel-button")).toBeNull();
  });

  it("treats a 404 cancel response as success (no toast, no Sentry)", () => {
    // 404 = session already promoted / not owned: not a real failure, the
    // UI just needs to resync. The destructive toast must NOT fire.
    errorResponse = { response: { status: 404 } };
    render(<QueueBadge sessionID="sess-x" />);
    fireEvent.click(screen.getByTestId("queue-cancel-button"));
    expect(toastMock).not.toHaveBeenCalled();
    expect(captureExceptionMock).not.toHaveBeenCalled();
  });

  it("shows the destructive toast and reports to Sentry on a real cancel error", () => {
    errorResponse = { response: { status: 500 } };
    render(<QueueBadge sessionID="sess-y" />);
    fireEvent.click(screen.getByTestId("queue-cancel-button"));
    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        variant: "destructive",
        title: "Could not cancel queued task",
      }),
    );
    expect(captureExceptionMock).toHaveBeenCalledTimes(1);
  });

  it("handles cancel errors with no response object (network error)", () => {
    // No `.response` on the error → the `status === 404` check should
    // be false, so we fall through to the toast + Sentry path.
    errorResponse = {};
    render(<QueueBadge sessionID="sess-z" />);
    fireEvent.click(screen.getByTestId("queue-cancel-button"));
    expect(toastMock).toHaveBeenCalled();
    expect(captureExceptionMock).toHaveBeenCalled();
  });
});
