import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useChatInput } from "../components/ChatInput/useChatInput";
import { useChatSession } from "../useChatSession";

// The first send of a new chat creates the session, and two layers used to
// report that one failure: useChatSession toasted "Please try again" and the
// composer toasted the server's reason, so a paywalled user saw two stacked
// red toasts and no way to upgrade. These run the real session hook into the
// real composer hook to pin the failure to one toast.

const toast = vi.hoisted(() => vi.fn());
const transports = vi.hoisted(() => ({ loaded: true }));
const billing = vi.hoisted(() => ({ enabled: true }));

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => billing.enabled,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toast(...args),
  useToast: () => ({ toast, dismiss: vi.fn() }),
}));

vi.mock(
  "@/app/api/__generated__/endpoints/chat/chat",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("@/app/api/__generated__/endpoints/chat/chat")
    >()),
    useGetV2ListChatTransports: () => ({
      data: !transports.loaded
        ? undefined
        : {
            status: 200,
            data: {
              transports: [
                {
                  auth_provider: "platform",
                  credential_id: null,
                  label: "AutoGPT Platform",
                  available: true,
                  default: true,
                },
              ],
            },
          },
      isError: false,
    }),
  }),
);

function Composer() {
  const { createSession } = useChatSession();
  const { value, handleChange, handleSubmit } = useChatInput({
    onSend: async () => {
      await createSession();
    },
  });
  return (
    <form onSubmit={handleSubmit}>
      <textarea aria-label="Message" value={value} onChange={handleChange} />
    </form>
  );
}

function answerSessionCreate(status: number, detail: string) {
  server.use(
    http.post("*/api/chat/sessions", () =>
      HttpResponse.json({ detail }, { status }),
    ),
  );
}

async function send(text: string) {
  render(<Composer />);
  const textarea = screen.getByRole<HTMLTextAreaElement>("textbox", {
    name: "Message",
  });
  fireEvent.change(textarea, { target: { value: text } });
  fireEvent.submit(textarea.closest("form")!);
  await waitFor(() => expect(toast).toHaveBeenCalled());
  return textarea;
}

afterEach(() => {
  server.resetHandlers();
  toast.mockReset();
  transports.loaded = true;
  billing.enabled = true;
});

describe("a refused first send", () => {
  it("offers plans once for a paywalled account and keeps the message", async () => {
    answerSessionCreate(
      402,
      "A subscription is required to use this feature. Upgrade to continue.",
    );
    const textarea = await send("draft my weekly report");

    expect(toast).toHaveBeenCalledOnce();
    const [paywall] = toast.mock.calls[0];
    expect(paywall).toMatchObject({
      title: "Sending needs an active plan",
      variant: "destructive",
    });
    render(
      <>
        {paywall.description}
        {paywall.action}
      </>,
    );
    expect(
      screen.getByText(
        "Choose one to keep chatting — your message is back in the composer.",
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: "See plans" }).getAttribute("href"),
    ).toBe("/settings/billing");
    expect(textarea.value).toBe("draft my weekly report");
  });

  it("offers no plans link where this install doesn't bill", async () => {
    billing.enabled = false;
    answerSessionCreate(
      402,
      "A subscription is required to use this feature. Upgrade to continue.",
    );
    await send("draft my weekly report");

    expect(toast).toHaveBeenCalledOnce();
    const [paywall] = toast.mock.calls[0];
    render(
      <>
        {paywall.description}
        {paywall.action}
      </>,
    );
    expect(screen.getByText(/Ask whoever runs this server/)).toBeDefined();
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
  });

  it("reports any other failure once, with the server's reason", async () => {
    answerSessionCreate(500, "Database unavailable");
    const textarea = await send("hello");

    expect(toast).toHaveBeenCalledOnce();
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't send message",
        variant: "destructive",
      }),
    );
    expect(toast.mock.calls[0][0].action).toBeUndefined();
    expect(textarea.value).toBe("hello");
  });

  it("says once that AI connections are still loading", async () => {
    transports.loaded = false;
    const textarea = await send("too quick");

    expect(toast).toHaveBeenCalledOnce();
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "AI connections are still loading" }),
    );
    expect(textarea.value).toBe("too quick");
  });
});
