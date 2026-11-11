import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { useOnboardingWizardStore } from "../../store";
import { AccessStep } from "../AccessStep/AccessStep";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: true,
    isUserLoading: false,
    user: { id: "user-1", email: "pat@example.com" },
  }),
}));

vi.mock("@/components/atoms/FadeIn/FadeIn", () => ({
  FadeIn: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

function signedUpWith(...providers: string[]) {
  server.use(
    http.get("*/api/auth/sign-in-providers", () =>
      HttpResponse.json({ providers }),
    ),
  );
}

beforeEach(() => {
  useOnboardingWizardStore.getState().reset();
  server.use(
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([
        {
          name: "microsoft",
          description: "Mail, calendar, files",
          supported_auth_types: ["oauth2", "device_code"],
        },
      ]),
    ),
    http.get("*/api/integrations/:provider/credentials", () =>
      HttpResponse.json([]),
    ),
  );
});

describe("AccessStep", () => {
  it("offers a Microsoft sign-up its Microsoft 365 connection", async () => {
    signedUpWith("microsoft");
    render(<AccessStep />);

    expect(
      await screen.findByRole("heading", {
        name: "Also connect Microsoft 365?",
      }),
    ).toBeDefined();
    // Signing up with Microsoft and connecting it are separate, and the step
    // says why it asks again.
    expect(
      screen.getByText(/Signing in with Microsoft only proves who you are\./),
    ).toBeDefined();
    expect(
      await screen.findByRole("button", { name: "Sign in with Microsoft" }),
    ).toBeDefined();
  });

  it("moves on to the next step when skipped", async () => {
    signedUpWith("microsoft");
    render(<AccessStep />);

    fireEvent.click(await screen.findByRole("button", { name: "Not now" }));

    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2),
    );
  });

  it("only offers to continue once the account is connected", async () => {
    signedUpWith("microsoft");
    server.use(
      http.get("*/api/integrations/:provider/credentials", ({ params }) =>
        HttpResponse.json(
          params.provider === "microsoft"
            ? [
                {
                  id: "work-id",
                  provider: "microsoft",
                  type: "oauth2",
                  title: "Pat Example (work account)",
                  username: "pat@company.example",
                  scopes: ["User.Read"],
                },
              ]
            : [],
        ),
      ),
      http.post("*/api/integrations/microsoft/check", () =>
        HttpResponse.json({
          verified: true,
          account_type: "work",
          message: "Connected.",
          admin_consent_url: null,
          capabilities: [],
        }),
      ),
    );
    render(<AccessStep />);

    fireEvent.click(await screen.findByRole("button", { name: "Continue" }));

    expect(screen.queryByRole("button", { name: "Not now" })).toBeNull();
    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2),
    );
  });

  it("offers nothing to an account that signed up with email", async () => {
    signedUpWith("credential");
    render(<AccessStep />);

    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: "Also connect Microsoft 365?" }),
      ).toBeNull(),
    );
  });
});
