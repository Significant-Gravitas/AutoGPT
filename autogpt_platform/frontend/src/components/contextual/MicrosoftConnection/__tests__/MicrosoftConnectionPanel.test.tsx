import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { MicrosoftConnectionPanel } from "../MicrosoftConnectionPanel";

vi.mock("@/lib/oauth-popup", async (original) => ({
  ...(await original<typeof import("@/lib/oauth-popup")>()),
  preOpenOAuthPopup: vi.fn(() => null),
}));

const TENANT = "dca42f33-178d-462f-9fea-f4e22af3a472";
const personal = {
  id: "personal-id",
  provider: "microsoft",
  type: "oauth2",
  title: "Same Person (personal account)",
  username: "same@company.example",
  scopes: ["User.Read", "Mail.ReadWrite"],
  service: "microsoft",
  microsoft_identity: {
    tenant_id: "9188040d-6c67-4c5b-b112-36a304b66dad",
    account_id: "personal-account",
    account_type: "personal",
  },
};
const work = {
  ...personal,
  id: "work-id",
  title: "Same Person (work account)",
  microsoft_identity: {
    tenant_id: TENANT,
    account_id: "work-account",
    account_type: "work",
  },
};

function capability(
  service: string,
  name: string,
  status: string,
  scopes = [],
) {
  return { service, name, status, scopes };
}

const workCheck = {
  verified: true,
  account_type: "work",
  message: "Connected.",
  admin_consent_url: `https://login.microsoftonline.com/${TENANT}/v2.0/adminconsent?client_id=app`,
  capabilities: [
    capability("mail", "Mail", "available"),
    capability("calendar", "Calendar", "available"),
    capability("files", "Files", "available"),
    {
      ...capability("teams", "Teams chats", "permission_needed"),
      scopes: ["Chat.ReadWrite"],
    },
    capability("copilot", "Microsoft 365 Copilot", "admin_required"),
    capability("work_iq", "Work IQ", "setup_required"),
  ],
};

const WORK_IQ = "api://workiq.svc.cloud.microsoft/WorkIQAgent.Ask";

let credentials: unknown[] = [];
let loginRequests: URL[] = [];
let workCheckOverride: unknown = null;

beforeEach(() => {
  credentials = [];
  loginRequests = [];
  workCheckOverride = null;
  server.use(
    http.get("*/api/integrations/:provider/credentials", ({ params }) =>
      HttpResponse.json(params.provider === "microsoft" ? credentials : []),
    ),
    http.get("*/api/integrations/:provider/login", ({ request }) => {
      loginRequests.push(new URL(request.url));
      return HttpResponse.json(
        {
          detail: {
            reason: "admin_consent_required",
            message: "Your Microsoft administrator needs to approve access.",
            error_code: "AADSTS90094",
          },
        },
        { status: 400 },
      );
    }),
    http.post("*/api/integrations/microsoft/check", async ({ request }) => {
      const body = (await request.json()) as { credential_id: string };
      return HttpResponse.json(
        body.credential_id === "work-id"
          ? (workCheckOverride ?? workCheck)
          : {
              verified: true,
              account_type: "personal",
              message: "Connected.",
              admin_consent_url: null,
              capabilities: [
                capability("mail", "Mail", "available"),
                capability("teams", "Teams chats", "work_account_only"),
              ],
            },
      );
    }),
  );
});

describe("Microsoft connection", () => {
  it("offers browser sign-in when the server can do it", async () => {
    render(<MicrosoftConnectionPanel browserAvailable onDone={vi.fn()} />);

    expect(
      await screen.findByRole("button", { name: "Sign in with Microsoft" }),
    ).toBeTruthy();
  });

  it("falls back to a sign-in code when the server has no app secret", async () => {
    render(
      <MicrosoftConnectionPanel browserAvailable={false} onDone={vi.fn()} />,
    );

    expect(
      await screen.findByRole("button", { name: "Connect Microsoft" }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "Sign in with Microsoft" }),
    ).toBeNull();
  });

  it("clears the last error when signing in again", async () => {
    render(<MicrosoftConnectionPanel browserAvailable onDone={vi.fn()} />);
    const signIn = await screen.findByRole("button", {
      name: "Sign in with Microsoft",
    });
    await waitFor(() => expect(signIn.hasAttribute("disabled")).toBe(false));
    fireEvent.click(signIn);
    expect(
      await screen.findByText(
        "Your Microsoft administrator needs to approve access.",
      ),
    ).toBeTruthy();

    let answer = () => {};
    const answered = new Promise<void>((resolve) => {
      answer = resolve;
    });
    server.use(
      http.get("*/api/integrations/:provider/login", async () => {
        await answered;
        return HttpResponse.json({ detail: "Try later" }, { status: 503 });
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() =>
      expect(
        screen.queryByText(
          "Your Microsoft administrator needs to approve access.",
        ),
      ).toBeNull(),
    );
    answer();
    expect(
      await screen.findByText(/Microsoft sign-in did not finish/),
    ).toBeTruthy();
  });

  it("offers a skip only until an account is connected", async () => {
    const onSkip = vi.fn();
    const { unmount } = render(
      <MicrosoftConnectionPanel
        browserAvailable
        onDone={vi.fn()}
        onSkip={onSkip}
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Not now" }));
    expect(onSkip).toHaveBeenCalledOnce();
    unmount();

    credentials = [work];
    render(
      <MicrosoftConnectionPanel
        browserAvailable
        onDone={vi.fn()}
        onSkip={onSkip}
      />,
    );
    expect(await screen.findByRole("button", { name: "Done" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Not now" })).toBeNull();
  });

  it("shows every service's status and asks only for what is missing", async () => {
    credentials = [work];
    render(<MicrosoftConnectionPanel browserAvailable onDone={vi.fn()} />);

    const list = await screen.findByRole("list", {
      name: "Microsoft services",
    });
    expect(within(list).getByText("Needs your permission")).toBeTruthy();
    expect(
      within(list).getByText("Needs your Microsoft admin's approval"),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Allow Teams chats" }));

    await waitFor(() => expect(loginRequests).toHaveLength(1));
    expect(loginRequests[0].searchParams.get("scopes")).toBe("Chat.ReadWrite");
    expect(loginRequests[0].searchParams.get("credential_id")).toBe("work-id");
    expect(
      await screen.findByText(
        "Your Microsoft administrator needs to approve access.",
      ),
    ).toBeTruthy();
  });

  it("clears a failed permission's error when checking again", async () => {
    credentials = [work];
    render(<MicrosoftConnectionPanel browserAvailable onDone={vi.fn()} />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Allow Teams chats" }),
    );
    expect(
      await screen.findByText(
        "Your Microsoft administrator needs to approve access.",
      ),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() =>
      expect(
        screen.queryByText(
          "Your Microsoft administrator needs to approve access.",
        ),
      ).toBeNull(),
    );
  });

  it("leaves Work IQ to its own Allow", async () => {
    credentials = [work];
    workCheckOverride = {
      ...workCheck,
      capabilities: [
        {
          ...capability("teams", "Teams chats", "permission_needed"),
          scopes: ["Chat.ReadWrite"],
        },
        {
          ...capability("work_iq", "Work IQ", "permission_needed"),
          scopes: [WORK_IQ],
        },
      ],
    };
    render(<MicrosoftConnectionPanel browserAvailable onDone={vi.fn()} />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Allow everything listed" }),
    );
    await waitFor(() => expect(loginRequests).toHaveLength(1));
    expect(loginRequests[0].searchParams.get("scopes")).toBe("Chat.ReadWrite");
    expect(screen.getByRole("button", { name: "Allow Work IQ" })).toBeTruthy();
  });

  it("gives Work IQ its own sign-in code", async () => {
    credentials = [work];
    workCheckOverride = {
      ...workCheck,
      capabilities: [
        {
          ...capability("work_iq", "Work IQ", "permission_needed"),
          scopes: [WORK_IQ],
        },
      ],
    };
    render(
      <MicrosoftConnectionPanel browserAvailable={false} onDone={vi.fn()} />,
    );

    expect(
      await screen.findByRole("button", { name: "Connect Work IQ" }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "Connect Microsoft" }),
    ).toBeNull();
  });

  it("lets an admin approve everything for the organization", async () => {
    credentials = [work];
    render(<MicrosoftConnectionPanel browserAvailable onDone={vi.fn()} />);

    const approve = await screen.findByRole("link", {
      name: "Approve for my whole organization",
    });
    expect(approve.getAttribute("href")).toBe(workCheck.admin_consent_url);
    expect(
      (
        await screen.findByRole("link", { name: "Turn on Work IQ (admins)" })
      ).getAttribute("href"),
    ).toBe(
      `https://login.microsoftonline.com/${TENANT}/adminconsent?client_id=fdcc1f02-fc51-4226-8753-f668596af7f7`,
    );
  });

  it("keeps same-email accounts separate", async () => {
    credentials = [personal, work];
    const onDone = vi.fn();
    render(<MicrosoftConnectionPanel browserAvailable onDone={onDone} />);

    const picker = await screen.findByRole("combobox", {
      name: "Microsoft account",
    });
    expect(
      await screen.findByText("Work or school accounts only"),
    ).toBeTruthy();

    fireEvent.click(picker);
    expect(await screen.findAllByRole("option")).toHaveLength(2);
    fireEvent.click(
      screen.getByRole("option", { name: "Same Person (work account)" }),
    );
    expect(
      await screen.findByRole("link", {
        name: "Approve for my whole organization",
      }),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(onDone).toHaveBeenCalledWith(
      expect.objectContaining({ id: "work-id" }),
    );
  });
});
