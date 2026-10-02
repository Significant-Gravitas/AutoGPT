import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import userEvent from "@testing-library/user-event";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import SignupPage from "../page";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockSignupAction = vi.hoisted(() => vi.fn());
const capture = vi.hoisted(() => vi.fn());
const setMarketingOptOutFlag = vi.hoisted(() => vi.fn());
const navigation = vi.hoisted(() => ({
  searchParams: new URLSearchParams(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => navigation.searchParams,
  usePathname: () => "/signup",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("../actions", () => ({
  signup: mockSignupAction,
}));

vi.mock("posthog-js", () => ({ default: { capture } }));

vi.mock("@/services/analytics/marketing-opt-out-cookie", () => ({
  setMarketingOptOutFlag,
}));

const PROVIDER_LOGIN_URL = "/api/auth/login/with-provider";

function fillValidForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: "new@example.com" },
  });
  fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
    target: { value: "validpassword123" },
  });
  fireEvent.change(
    screen.getByLabelText("Confirm Password", { selector: "input" }),
    { target: { value: "validpassword123" } },
  );
}

function getLegalLine() {
  return screen.getByText(/By continuing you agree to our/);
}

function stubProviderFetch() {
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async () => new Response(JSON.stringify({})));
}

function providerFetchCallOrder(
  fetchSpy: ReturnType<typeof stubProviderFetch>,
) {
  const index = fetchSpy.mock.calls.findIndex(
    ([input]) => input === PROVIDER_LOGIN_URL,
  );
  return fetchSpy.mock.invocationCallOrder[index];
}

describe("SignupPage", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: null,
      isUserLoading: false,
      isLoggedIn: false,
    });
    mockSignupAction.mockReset();
    mockSignupAction.mockResolvedValue({
      success: false,
      error: "user_already_exists",
    });
    capture.mockReset();
    setMarketingOptOutFlag.mockReset();
    navigation.searchParams = new URLSearchParams();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  test("shows existing user feedback from signup action", async () => {
    render(<SignupPage />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "existing@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "validpassword123" },
    });
    fireEvent.change(
      screen.getByLabelText("Confirm Password", { selector: "input" }),
      {
        target: { value: "validpassword123" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: "Sign up" }));

    await waitFor(() => {
      expect(mockSignupAction).toHaveBeenCalledWith(
        "existing@example.com",
        "validpassword123",
        "validpassword123",
        false,
      );
    });

    expect(
      await screen.findByText("User with this email already exists"),
    ).toBeDefined();
  });

  test("signs up without a terms checkbox", async () => {
    render(<SignupPage />);

    expect(screen.queryByRole("checkbox")).toBeNull();
    expect(screen.queryByText(/I agree to the/)).toBeNull();

    fillValidForm();
    fireEvent.click(screen.getByRole("button", { name: "Sign up" }));

    await waitFor(() => expect(mockSignupAction).toHaveBeenCalledTimes(1));
  });

  test("puts the Log in link under the title, keeping ?next=", () => {
    navigation.searchParams = new URLSearchParams({
      next: "/library?tab=runs",
    });

    render(<SignupPage />);

    const logInLinks = screen.getAllByRole("link", { name: "Log in" });
    expect(logInLinks).toHaveLength(1);
    const [logIn] = logInLinks;
    expect(logIn.getAttribute("href")).toBe(
      "/login?next=%2Flibrary%3Ftab%3Druns",
    );
    expect(logIn.parentElement?.textContent).toBe("Already a member? Log in");

    const header = screen.getByRole("heading", {
      name: "Create your account",
    }).parentElement;
    expect(header?.textContent).toBe(
      "Create your accountAlready a member? Log in",
    );
    expect(header?.nextElementSibling?.tagName).toBe("FORM");
  });

  test("shows the legal line with the terms and privacy links", () => {
    render(<SignupPage />);

    expect(getLegalLine().textContent).toBe(
      "By continuing you agree to our Terms of Use and Privacy Policy. We may email you product updates and offers; opt out.",
    );

    const terms = screen.getByRole("link", { name: "Terms of Use" });
    expect(terms.getAttribute("href")).toBe(
      "https://agpt.co/legal/platform-terms-of-use",
    );
    expect(terms.getAttribute("target")).toBe("_blank");
    expect(terms.getAttribute("rel")).toBe("noopener noreferrer");

    const privacy = screen.getByRole("link", { name: "Privacy Policy" });
    expect(privacy.getAttribute("href")).toBe(
      "https://agpt.co/legal/platform-privacy-policy",
    );
    expect(privacy.getAttribute("target")).toBe("_blank");
    expect(privacy.getAttribute("rel")).toBe("noopener noreferrer");
  });

  test("renders the legal line right after the form when there is no Google button", () => {
    render(<SignupPage />);

    expect(
      screen.queryByRole("button", { name: /continue with google/i }),
    ).toBeNull();
    const previous = getLegalLine().previousElementSibling;
    expect(previous?.tagName).toBe("FORM");
    expect(previous?.lastElementChild?.textContent).toBe("Sign up");
  });

  test("renders the legal line right after the Google button in cloud", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");

    render(<SignupPage />);

    expect(getLegalLine().previousElementSibling).toBe(
      screen.getByRole("button", { name: /continue with google/i }),
    );
  });

  test("opting out swaps the sentence, keeps focus on the toggle and submits the opt-out", async () => {
    const user = userEvent.setup();
    render(<SignupPage />);

    const toggle = screen.getByRole("button", { name: "opt out" });
    await user.click(toggle);

    expect(getLegalLine().textContent).toBe(
      "By continuing you agree to our Terms of Use and Privacy Policy. You won't get marketing emails. Undo.",
    );
    expect(screen.getByRole("button", { name: "Undo" })).toBe(toggle);
    expect(document.activeElement).toBe(toggle);
    expect(
      screen
        .getByText("You won't get marketing emails.")
        .closest("[aria-live]")
        ?.getAttribute("aria-live"),
    ).toBe("polite");
    expect(screen.queryByText(/We may email you/)).toBeNull();

    fillValidForm();
    fireEvent.click(screen.getByRole("button", { name: "Sign up" }));

    await waitFor(() => {
      expect(mockSignupAction).toHaveBeenCalledWith(
        "new@example.com",
        "validpassword123",
        "validpassword123",
        true,
      );
    });
  });

  test("Undo restores the copy and submits without an opt-out", async () => {
    const user = userEvent.setup();
    render(<SignupPage />);

    const toggle = screen.getByRole("button", { name: "opt out" });
    await user.click(toggle);
    await user.click(screen.getByRole("button", { name: "Undo" }));

    expect(getLegalLine().textContent).toBe(
      "By continuing you agree to our Terms of Use and Privacy Policy. We may email you product updates and offers; opt out.",
    );
    expect(screen.getByRole("button", { name: "opt out" })).toBe(toggle);
    expect(document.activeElement).toBe(toggle);

    fillValidForm();
    fireEvent.click(screen.getByRole("button", { name: "Sign up" }));

    await waitFor(() => {
      expect(mockSignupAction).toHaveBeenCalledWith(
        "new@example.com",
        "validpassword123",
        "validpassword123",
        false,
      );
    });
  });

  test("captures the opt-out once with no properties and never on Undo", async () => {
    const user = userEvent.setup();
    render(<SignupPage />);

    expect(capture).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "opt out" }));
    expect(capture.mock.calls).toEqual([["signup_marketing_opt_out"]]);

    await user.click(screen.getByRole("button", { name: "Undo" }));
    expect(capture).toHaveBeenCalledTimes(1);
  });

  test("Google signup carries an opt-out across the redirect, set before leaving", async () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    const fetchSpy = stubProviderFetch();
    const user = userEvent.setup();
    render(<SignupPage />);

    await user.click(screen.getByRole("button", { name: "opt out" }));
    await user.click(
      screen.getByRole("button", { name: /continue with google/i }),
    );

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        PROVIDER_LOGIN_URL,
        expect.objectContaining({ method: "POST" }),
      );
    });
    expect(setMarketingOptOutFlag.mock.calls).toEqual([[true]]);
    expect(setMarketingOptOutFlag.mock.invocationCallOrder[0]).toBeLessThan(
      providerFetchCallOrder(fetchSpy),
    );
  });

  test("Google signup clears any stale opt-out when the user did not opt out", async () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    const fetchSpy = stubProviderFetch();
    const user = userEvent.setup();
    render(<SignupPage />);

    await user.click(
      screen.getByRole("button", { name: /continue with google/i }),
    );

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        PROVIDER_LOGIN_URL,
        expect.objectContaining({ method: "POST" }),
      );
    });
    expect(setMarketingOptOutFlag.mock.calls).toEqual([[false]]);
    expect(setMarketingOptOutFlag.mock.invocationCallOrder[0]).toBeLessThan(
      providerFetchCallOrder(fetchSpy),
    );
    expect(capture).not.toHaveBeenCalled();
  });

  test("does not link to the demo tour", () => {
    render(<SignupPage />);

    expect(screen.getByRole("button", { name: "Sign up" })).toBeDefined();
    expect(screen.queryByText(/watch the demo/i)).toBeNull();
    expect(
      screen
        .queryAllByRole("link")
        .some((link) => link.getAttribute("href")?.startsWith("/tour")),
    ).toBe(false);
  });

  test("does not server-render an interactive form before auth initializes", () => {
    const markup = renderToStaticMarkup(<SignupPage />);

    expect(markup).not.toContain('id="password"');
    expect(markup).not.toContain('type="submit"');
  });

  test("preserves form input during a background auth refresh", () => {
    const { rerender } = render(<SignupPage />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "draft@example.com" },
    });

    mockUseAuth.mockReturnValue({
      user: null,
      isUserLoading: true,
      isLoggedIn: false,
    });
    rerender(<SignupPage />);

    expect((screen.getByLabelText("Email") as HTMLInputElement).value).toBe(
      "draft@example.com",
    );
  });
});
