import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { getPostV2NotifyImpersonationStartMockHandler200 } from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { getGetV1GetUserCreditsMockHandler } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { ImpersonationState } from "@/lib/impersonation";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import { AdminImpersonationPanel } from "../AdminImpersonationPanel";

const NOTIFY_URL =
  "http://localhost:3000/api/proxy/api/admin/impersonation/notify";
const VALID_UUID = "2e7ea138-2097-425d-9cad-c660f29cc251";

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, useToast: () => ({ toast: toastMock }) };
});

const reloadMock = vi.fn();

beforeEach(() => {
  toastMock.mockClear();
  reloadMock.mockClear();
  sessionStorage.clear();
  Object.defineProperty(window.location, "reload", {
    configurable: true,
    value: reloadMock,
  });
  // Avoid persisting impersonation state across tests; assert reload instead.
  vi.spyOn(ImpersonationState, "set").mockImplementation(() => {});
  // The panel renders a live-credits demo; keep that query happy.
  server.use(getGetV1GetUserCreditsMockHandler());
});

afterEach(() => {
  vi.restoreAllMocks();
});

function typeUserId(value: string) {
  fireEvent.change(screen.getByLabelText(/user id to impersonate/i), {
    target: { value },
  });
}

function clickStart() {
  fireEvent.click(screen.getByRole("button", { name: /^start$/i }));
}

describe("AdminImpersonationPanel", () => {
  test("disables Start until a user id is entered", () => {
    render(<AdminImpersonationPanel />);
    const start = screen.getByRole("button", {
      name: /^start$/i,
    }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);

    typeUserId(VALID_UUID);
    expect(start.disabled).toBe(false);
  });

  test("rejects a non-UUID value with an inline error and no swap", async () => {
    render(<AdminImpersonationPanel />);
    typeUserId("not-a-uuid");
    clickStart();

    expect(await screen.findByText(/valid uuid format/i)).toBeDefined();
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("keeps the entered id and shows a destructive toast when the alert is blocked (502)", async () => {
    server.use(
      http.post(NOTIFY_URL, () => new HttpResponse(null, { status: 502 })),
    );

    render(<AdminImpersonationPanel />);
    const input = screen.getByLabelText(
      /user id to impersonate/i,
    ) as HTMLInputElement;
    typeUserId(VALID_UUID);
    clickStart();

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "destructive" }),
      );
    });
    // The fix: a blocked alert must not wipe the entered UUID.
    expect(input.value).toBe(VALID_UUID);
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("starts impersonation (reload) when the audit alert is delivered", async () => {
    server.use(
      getPostV2NotifyImpersonationStartMockHandler200({ alerted: true }),
    );

    render(<AdminImpersonationPanel />);
    typeUserId(VALID_UUID);
    clickStart();

    await waitFor(() => expect(reloadMock).toHaveBeenCalledTimes(1));
  });
});
