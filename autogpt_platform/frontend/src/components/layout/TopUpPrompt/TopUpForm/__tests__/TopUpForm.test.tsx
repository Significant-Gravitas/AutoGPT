import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  act,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@testing-library/react";
import { server } from "@/mocks/mock-server";
import { getPostV1RequestCreditTopUpMockHandler } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { HttpResponse, http } from "msw";

import { TopUpForm } from "../TopUpForm";

const routerPush = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useParams: () => ({}),
  usePathname: () => "/marketplace",
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: routerPush,
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

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

beforeEach(() => {
  routerPush.mockClear();
  useAuthStore.setState({ user: { id: "user-a" } as User });
});

afterEach(() => {
  useAuthStore.setState({ user: null });
});

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

  test("does not navigate to a checkout started by the previous identity", async () => {
    let releaseRequest: (() => void) | undefined;
    server.use(
      http.post("/api/proxy/api/credits", async () => {
        await new Promise<void>((resolve) => {
          releaseRequest = resolve;
        });
        return HttpResponse.json({
          checkout_url: "https://checkout.example/old-user",
        });
      }),
    );

    render(<TopUpForm />);
    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));
    await screen.findByRole("button", { name: /redirecting/i });
    await waitFor(() => expect(releaseRequest).toBeDefined());

    act(() => {
      useAuthStore.setState({ user: { id: "user-b" } as User });
    });
    releaseRequest?.();

    await screen.findByRole("button", { name: /top up/i });
    expect(routerPush).not.toHaveBeenCalled();
  });
});
