import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { useOnboardingWizardStore } from "../../store";
import { AccessStep } from "../AccessStep/AccessStep";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: true,
    isUserLoading: false,
    user: { id: "user-1", email: "pat@gmail.com" },
  }),
}));

vi.mock("@/components/atoms/FadeIn/FadeIn", () => ({
  FadeIn: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

// Google's own window, answered as if the person went through it.
vi.mock("@/lib/oauth-popup", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/oauth-popup")>()),
  preOpenOAuthPopup: vi.fn(() => null),
  openOAuthPopup: vi.fn(() => ({
    promise: Promise.resolve({ code: "google-code", state: "state-token" }),
    cleanup: { abort: vi.fn() },
    popupBlocked: false,
    fallbackBlocked: false,
  })),
}));

const SCOPE = "https://www.googleapis.com/auth/";
const GMAIL = ["gmail.readonly", "gmail.send", "gmail.modify", "gmail.labels"];
const DRIVE = ["drive.file"];
const CALENDAR = ["calendar", "calendar.readonly"];

function scopes(...names: string[]) {
  return names.map((name) => `${SCOPE}${name}`);
}

function googleAccount(granted: string[]) {
  return {
    id: "google-1",
    provider: "google",
    type: "oauth2",
    title: null,
    username: "pat@gmail.com",
    scopes: [...scopes("userinfo.email"), "openid", ...granted],
  };
}

// What Google grants when asked, and the connections saved so far.
let grants: string[] = [];
let saved: Record<string, unknown>[] = [];
let logins: URL[] = [];

beforeEach(() => {
  useOnboardingWizardStore.getState().reset();
  grants = scopes(...GMAIL, ...DRIVE, ...CALENDAR);
  saved = [];
  logins = [];
  server.use(
    http.get("*/api/auth/sign-in-providers", () =>
      HttpResponse.json({ providers: ["google"] }),
    ),
    http.get("*/api/integrations/:provider/credentials", ({ params }) =>
      HttpResponse.json(params.provider === "google" ? saved : []),
    ),
    http.get("*/api/integrations/google/login", ({ request }) => {
      logins.push(new URL(request.url));
      return HttpResponse.json({
        login_url: "https://accounts.google.com/o/oauth2/auth",
        state_token: "state-token",
      });
    }),
    http.post("*/api/integrations/google/callback", () => {
      const account = googleAccount(grants);
      saved = [account];
      return HttpResponse.json(account);
    }),
  );
});

function requested(login: URL) {
  return (login.searchParams.get("scopes") ?? "").split(",").sort();
}

function serviceRow(name: string) {
  const list = screen.getByRole("list", { name: "Google services" });
  const row = within(list).getByText(name).closest("li");
  if (!row) throw new Error(`No row for ${name}`);
  return within(row);
}

describe("Google access step", () => {
  it("offers a Google sign-up its Gmail, Drive and Calendar connection", async () => {
    render(<AccessStep />);

    expect(
      await screen.findByRole("heading", {
        name: "Also connect Gmail, Drive and Calendar?",
      }),
    ).toBeDefined();
    expect(
      screen.getByText(/Signing in with Google only proves who you are\./),
    ).toBeDefined();
    // Each service says what its permissions are for before any are asked.
    expect(
      await screen.findByText(
        "Lets Otto read, send, label and trash your email",
      ),
    ).toBeDefined();
    expect(
      screen.getByText("Lets Otto see, add and change your calendar events"),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Connect Google" }),
    ).toBeDefined();
    // Each row's status region exists before there's a status, so a screen
    // reader announces the first one when the account connects.
    const list = screen.getByRole("list", { name: "Google services" });
    expect(list.querySelectorAll('[aria-live="polite"]')).toHaveLength(3);
  });

  it("asks Google once for everything the Google blocks use", async () => {
    render(<AccessStep />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Connect Google" }),
    );

    await waitFor(() => expect(logins).toHaveLength(1));
    expect(requested(logins[0])).toEqual(
      scopes(...GMAIL, ...DRIVE, ...CALENDAR).sort(),
    );
    // With the metadata scope in a grant, Gmail returns headers only.
    expect(requested(logins[0])).not.toContain(`${SCOPE}gmail.metadata`);
    expect(logins[0].searchParams.get("credential_id")).toBeNull();

    expect(await screen.findByText("pat@gmail.com")).toBeDefined();
    expect(serviceRow("Gmail").getByText("Ready")).toBeDefined();
    expect(serviceRow("Google Drive").getByText("Ready")).toBeDefined();
    expect(serviceRow("Google Calendar").getByText("Ready")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Not now" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2),
    );
  });

  it("asks again only for what was left unticked on Google's screen", async () => {
    saved = [googleAccount(scopes(...GMAIL, ...DRIVE))];
    render(<AccessStep />);

    expect(
      await screen.findByRole("button", { name: "Allow Google Calendar" }),
    ).toBeDefined();
    expect(serviceRow("Gmail").getByText("Ready")).toBeDefined();
    expect(
      serviceRow("Google Calendar").getByText("Needs your permission"),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Allow Gmail" })).toBeNull();

    fireEvent.click(
      screen.getByRole("button", { name: "Allow Google Calendar" }),
    );

    await waitFor(() => expect(logins).toHaveLength(1));
    // Added to the same account, not a second connection.
    expect(requested(logins[0])).toEqual(scopes(...CALENDAR).sort());
    expect(logins[0].searchParams.get("credential_id")).toBe("google-1");
    await waitFor(() =>
      expect(serviceRow("Google Calendar").getByText("Ready")).toBeDefined(),
    );
  });

  it("shows the account that allows the most when several are connected", async () => {
    const work = { ...googleAccount(scopes(...GMAIL, ...DRIVE, ...CALENDAR)) };
    saved = [
      { ...googleAccount([]), id: "google-0", username: "pat@old.example" },
      { ...work, username: "pat@company.example" },
    ];
    render(<AccessStep />);

    expect(await screen.findByText("pat@company.example")).toBeDefined();
    expect(serviceRow("Google Calendar").getByText("Ready")).toBeDefined();
    expect(screen.getByRole("button", { name: "Continue" })).toBeDefined();
  });

  it("asks for every service left in one go", async () => {
    // Connected without allowing anything, e.g. everything unticked.
    saved = [googleAccount([])];
    render(<AccessStep />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Allow everything listed" }),
    );

    await waitFor(() => expect(logins).toHaveLength(1));
    expect(requested(logins[0])).toEqual(
      scopes(...GMAIL, ...DRIVE, ...CALENDAR).sort(),
    );
    expect(logins[0].searchParams.get("credential_id")).toBe("google-1");
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Allow everything listed" }),
      ).toBeNull(),
    );
    expect(serviceRow("Gmail").getByText("Ready")).toBeDefined();
  });

  it("never takes another kind of Google credential for the account", async () => {
    saved = [
      {
        id: "key-1",
        provider: "google",
        type: "api_key",
        title: "Old key",
        username: null,
        scopes: null,
      },
    ];
    render(<AccessStep />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Connect Google" }),
    );

    await waitFor(() => expect(logins).toHaveLength(1));
    expect(logins[0].searchParams.get("credential_id")).toBeNull();
    expect(screen.queryByText("Old key")).toBeNull();
  });

  it("offers to check again, not to connect, when the lookup fails", async () => {
    // "Not connected" would offer a first connection to an account that may
    // already have one.
    let failing = true;
    server.use(
      http.get("*/api/integrations/:provider/credentials", ({ params }) =>
        failing && params.provider === "google"
          ? HttpResponse.json({ detail: "down" }, { status: 500 })
          : HttpResponse.json(params.provider === "google" ? saved : []),
      ),
    );
    render(<AccessStep />);

    expect(
      await screen.findByText(
        "We had the following error when retrieving your Google connection:",
      ),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Connect Google" })).toBeNull();

    failing = false;
    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));

    expect(
      await screen.findByRole("button", { name: "Connect Google" }),
    ).toBeDefined();
  });

  it("can still move on after a connect fails", async () => {
    server.use(
      http.get("*/api/integrations/google/login", () =>
        HttpResponse.json({ detail: "not configured" }, { status: 501 }),
      ),
    );
    render(<AccessStep />);

    const connect = await screen.findByRole("button", {
      name: "Connect Google",
    });
    fireEvent.click(connect);

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Connect Google" }),
      ).not.toHaveProperty("disabled", true),
    );
    fireEvent.click(screen.getByRole("button", { name: "Not now" }));
    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2),
    );
  });

  it("moves on to the next step when skipped", async () => {
    render(<AccessStep />);

    fireEvent.click(await screen.findByRole("button", { name: "Not now" }));

    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2),
    );
    expect(logins).toHaveLength(0);
  });
});
