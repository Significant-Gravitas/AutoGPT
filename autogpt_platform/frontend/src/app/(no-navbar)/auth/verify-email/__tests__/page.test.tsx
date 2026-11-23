import { render, screen } from "@/tests/integrations/test-utils";
import { UnsecuredJWT } from "jose";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ heldLink: undefined as string | undefined }));

const RedirectSignal = vi.hoisted(
  () =>
    class RedirectSignal {
      constructor(readonly url: string) {}
    },
);

vi.mock("next/headers", () => ({
  cookies: vi.fn(async () => ({
    get: vi.fn((name: string) =>
      name === "email_verification_link" && mocks.heldLink
        ? { value: mocks.heldLink }
        : undefined,
    ),
  })),
}));
// The shared mock's router hooks, which the test providers need, plus redirect.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  usePathname: () => "/auth/verify-email",
  useSearchParams: () => new URLSearchParams(),
  redirect: vi.fn((url: string) => {
    throw new RedirectSignal(url);
  }),
}));

import VerifyEmailPage from "../page";

describe("the email confirm page", () => {
  beforeEach(() => {
    mocks.heldLink = undefined;
  });

  it("names the address the link verifies and offers to verify and sign in", async () => {
    const token = new UnsecuredJWT({ email: "jamie@example.com" }).encode();
    mocks.heldLink = new URLSearchParams({ token }).toString();

    const { container } = render(await VerifyEmailPage());

    expect(
      screen.getByRole("heading", { name: "Confirm your email" }),
    ).toBeDefined();
    expect(screen.getByText("jamie@example.com")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Verify and sign in" }),
    ).toBeDefined();
    expect(container.querySelector("svg")?.getAttribute("aria-hidden")).toBe(
      "true",
    );
  });

  it("sends a visit without a waiting link to the login page's expired notice", async () => {
    const signal = await VerifyEmailPage().catch((thrown: unknown) => thrown);

    expect(signal).toEqual(
      new RedirectSignal("/login?email_verification=expired"),
    );
  });
});
