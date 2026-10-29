import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn() }),
  useToastOnFail: () => () => {},
}));

import { InlineHostScopedForm } from "../InlineHostScopedForm";

const CREATE_CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/:provider/credentials";

const PREFILLED_HOST = "api.example.com:8443";

async function findConnectButton() {
  return (await screen.findByRole("button", {
    name: "Connect",
  })) as HTMLButtonElement;
}

async function fillOneHeader() {
  await userEvent.type(
    await screen.findByLabelText("Header name"),
    "Authorization",
  );
  await userEvent.type(
    await screen.findByLabelText("Header value"),
    "Bearer t",
  );
}

describe("InlineHostScopedForm with a prefilled host", () => {
  it("shows the host the block calls, port included, and locks it", async () => {
    render(
      <InlineHostScopedForm
        provider="http"
        host={PREFILLED_HOST}
        onSuccess={vi.fn()}
      />,
    );

    const hostField = (await screen.findByLabelText(
      "Host",
    )) as HTMLInputElement;
    expect(hostField.value).toBe(PREFILLED_HOST);
    expect(hostField.readOnly).toBe(true);
  });

  // The prefilled host is never edited, so a gate reading react-hook-form's
  // isValid — false until the first change — would leave Connect dead.
  it("enables Connect without touching the prefilled host", async () => {
    render(
      <InlineHostScopedForm
        provider="http"
        host={PREFILLED_HOST}
        onSuccess={vi.fn()}
      />,
    );

    expect((await findConnectButton()).disabled).toBe(true);

    await fillOneHeader();

    await waitFor(async () =>
      expect((await findConnectButton()).disabled).toBe(false),
    );
  });

  it("saves the prefilled host verbatim", async () => {
    const posted = vi.fn();
    server.use(
      http.post(CREATE_CREDENTIALS_URL, async ({ request }) => {
        posted(await request.json());
        return HttpResponse.json({
          id: "cred-1",
          provider: "http",
          type: "host_scoped",
          title: PREFILLED_HOST,
        });
      }),
    );
    const onSuccess = vi.fn();

    render(
      <InlineHostScopedForm
        provider="http"
        host={PREFILLED_HOST}
        onSuccess={onSuccess}
      />,
    );

    await fillOneHeader();
    await userEvent.click(await findConnectButton());

    await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    expect(posted).toHaveBeenCalledWith(
      expect.objectContaining({ host: PREFILLED_HOST }),
    );
  });
});
