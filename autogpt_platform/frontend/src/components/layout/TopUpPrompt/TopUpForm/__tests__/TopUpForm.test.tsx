import { describe, expect, test } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getPostV1RequestCreditTopUpMockHandler } from "@/app/api/__generated__/endpoints/credits/credits.msw";

import { TopUpForm } from "../TopUpForm";

function captureTopUpRequest() {
  let body: unknown = null;
  server.use(
    getPostV1RequestCreditTopUpMockHandler(
      async (info: { request: Request }) => {
        body = await info.request.json();
      },
    ),
  );
  return { getBody: () => body };
}

describe("TopUpForm", () => {
  test("submits the entered amount converted to cents", async () => {
    const { getBody } = captureTopUpRequest();

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    await waitFor(() => expect(getBody()).toEqual({ credit_amount: 1500 }));
  });

  test("rejects amounts below the $5 minimum without calling the API", async () => {
    const { getBody } = captureTopUpRequest();

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "3" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    expect(await screen.findByText(/Top-ups start at \$5/i)).toBeDefined();
    expect(getBody()).toBeNull();
  });

  test("disables the submit button while the checkout request is in flight", async () => {
    // Hold the request open so the in-flight loading state is observable.
    server.use(
      getPostV1RequestCreditTopUpMockHandler(
        () => new Promise((resolve) => setTimeout(resolve, 300)),
      ),
    );

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    const button = await screen.findByRole("button", { name: /redirecting/i });
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });
});
