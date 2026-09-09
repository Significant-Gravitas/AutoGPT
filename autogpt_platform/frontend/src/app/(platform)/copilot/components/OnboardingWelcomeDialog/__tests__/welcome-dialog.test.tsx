import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.fn();
vi.mock("posthog-js", () => ({
  default: { capture: (...a: unknown[]) => capture(...a) },
}));

// All off by default, so every existing assertion keeps describing the
// flag-off deck; the team tests below turn both flags on.
const flags = vi.hoisted(() => ({ current: {} as Record<string, boolean> }));
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flags.current[flag] ?? false,
  };
});

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastSpy(...args),
  useToast: () => ({ toast: toastSpy }),
  useToastOnFail: () => () => {},
}));

import { OnboardingWelcomeDialog } from "../OnboardingWelcomeDialog";

const STEP_URL = "http://localhost:3000/api/proxy/api/onboarding/step";
const PROVIDERS_URL =
  "http://localhost:3000/api/proxy/api/integrations/providers";
const CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/credentials";
const RECOMMENDED_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/recommended-providers";

// Every step completion the dialog posts, with the step name it claimed.
function recordCompletedSteps(status = 200) {
  const steps: (string | null)[] = [];
  server.use(
    http.post(STEP_URL, ({ request }) => {
      steps.push(new URL(request.url).searchParams.get("step"));
      return new HttpResponse(null, { status });
    }),
  );
  return steps;
}

function stubConnectPanelEndpoints() {
  server.use(
    http.get(PROVIDERS_URL, () =>
      HttpResponse.json([
        {
          name: "github",
          description: "Code host",
          supported_auth_types: ["oauth2"],
        },
      ]),
    ),
    http.get(CREDENTIALS_URL, () => HttpResponse.json([])),
    http.get(RECOMMENDED_URL, () =>
      HttpResponse.json({ ready: true, providers: [] }),
    ),
  );
}

async function advanceToCard(index: number) {
  const user = userEvent.setup();
  for (let i = 0; i < index; i++) {
    await user.click(screen.getByRole("button", { name: "Next" }));
  }
  return user;
}

beforeEach(() => {
  capture.mockClear();
  toastSpy.mockClear();
  flags.current = {};
});

describe("OnboardingWelcomeDialog — deck", () => {
  it("renders the first card with no way back", async () => {
    render(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);

    expect(await screen.findByText("Meet AutoPilot.")).toBeDefined();
    expect(
      screen.getByText(
        "It does the work. Ask once, or put it on a schedule. It delivers while you do something else.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Next" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Skip" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Previous card" })).toBeNull();
  });

  it("walks forward and back through the deck", async () => {
    render(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);
    const user = await advanceToCard(1);

    expect(
      await screen.findByText("It works inside your tools."),
    ).toBeDefined();
    expect(capture).toHaveBeenCalledWith("capability_card_viewed", {
      card_index: 1,
    });

    await user.click(screen.getByRole("button", { name: "Previous card" }));

    expect(await screen.findByText("Meet AutoPilot.")).toBeDefined();
  });

  it("renders nothing and completes no step while closed", async () => {
    const steps = recordCompletedSteps();

    render(<OnboardingWelcomeDialog isOpen={false} onClose={vi.fn()} />);

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByText("Meet AutoPilot.")).toBeNull();
    expect(steps).toEqual([]);
  });
});

describe("OnboardingWelcomeDialog — completion", () => {
  it("records CAPABILITY_CARDS and closes once the last card is finished", async () => {
    const steps = recordCompletedSteps();
    const onClose = vi.fn();

    render(<OnboardingWelcomeDialog isOpen onClose={onClose} />);
    const user = await advanceToCard(3);

    expect(await screen.findByText("It remembers everything.")).toBeDefined();
    await user.click(screen.getByRole("button", { name: "Meet AutoPilot" }));

    expect(onClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(steps).toEqual(["CAPABILITY_CARDS"]));
    expect(capture).toHaveBeenCalledWith("capability_cards_completed", {
      card_index: 3,
    });
  });

  it("records the same step when skipped, tagged with the card the user quit on", async () => {
    const steps = recordCompletedSteps();
    const onClose = vi.fn();

    render(<OnboardingWelcomeDialog isOpen onClose={onClose} />);
    const user = await advanceToCard(1);
    await screen.findByText("It works inside your tools.");

    await user.click(screen.getByRole("button", { name: "Skip" }));

    expect(onClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(steps).toEqual(["CAPABILITY_CARDS"]));
    expect(capture).toHaveBeenCalledWith("capability_cards_skipped", {
      card_index: 1,
    });
  });

  it("treats Escape as a skip", async () => {
    const steps = recordCompletedSteps();
    const onClose = vi.fn();

    render(<OnboardingWelcomeDialog isOpen onClose={onClose} />);
    await screen.findByText("Meet AutoPilot.");

    await userEvent.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(steps).toEqual(["CAPABILITY_CARDS"]));
    expect(capture).toHaveBeenCalledWith("capability_cards_skipped", {
      card_index: 0,
    });
  });

  it("warns the user when the progress could not be saved", async () => {
    recordCompletedSteps(500);

    render(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);
    await screen.findByText("Meet AutoPilot.");

    await userEvent.click(screen.getByRole("button", { name: "Skip" }));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith({
        title: "Could not save your onboarding progress",
        description: "You may see this introduction again next time.",
        variant: "destructive",
      }),
    );
  });
});

describe("OnboardingWelcomeDialog — connect tools CTA", () => {
  it("swaps the card for the provider picker without closing the dialog", async () => {
    stubConnectPanelEndpoints();
    const steps = recordCompletedSteps();
    const onClose = vi.fn();

    render(<OnboardingWelcomeDialog isOpen onClose={onClose} />);
    const user = await advanceToCard(1);
    await screen.findByText("It works inside your tools.");

    await user.click(
      screen.getByRole("button", { name: "Connect your tools" }),
    );

    expect(
      await screen.findByRole("heading", { name: "Connect your tools" }),
    ).toBeDefined();
    expect(screen.queryByText("It works inside your tools.")).toBeNull();
    // Connecting tools must never end the introduction.
    expect(onClose).not.toHaveBeenCalled();
    expect(steps).toEqual([]);

    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(
      await screen.findByText("It works inside your tools."),
    ).toBeDefined();
  });

  it("steps back out of the picker on Escape instead of ending onboarding", async () => {
    stubConnectPanelEndpoints();
    const steps = recordCompletedSteps();
    const onClose = vi.fn();

    render(<OnboardingWelcomeDialog isOpen onClose={onClose} />);
    const user = await advanceToCard(1);
    await user.click(
      screen.getByRole("button", { name: "Connect your tools" }),
    );
    await screen.findByRole("heading", { name: "Connect your tools" });

    await user.keyboard("{Escape}");

    // Back to the deck, with the introduction still running.
    expect(
      await screen.findByText("It works inside your tools."),
    ).toBeDefined();
    expect(onClose).not.toHaveBeenCalled();
    expect(steps).toEqual([]);
    expect(capture).not.toHaveBeenCalledWith(
      "capability_cards_skipped",
      expect.anything(),
    );

    // Escape is a skip again once the picker is gone.
    await user.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("advances to the next card when the picker's Next is used", async () => {
    stubConnectPanelEndpoints();

    render(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);
    const user = await advanceToCard(1);
    await screen.findByText("It works inside your tools.");
    await user.click(
      screen.getByRole("button", { name: "Connect your tools" }),
    );
    await screen.findByRole("heading", { name: "Connect your tools" });

    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByText("It learns how you operate.")).toBeDefined();
  });
});

describe("OnboardingWelcomeDialog — the team deck", () => {
  beforeEach(() => {
    flags.current = {
      "onboarding-expert-team": true,
      "hire-experts": true,
    };
  });

  it("frames AutoPilot as the Head of AI that builds a team", async () => {
    render(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);

    expect(await screen.findByText("Meet your Head of AI.")).toBeDefined();
    expect(
      screen.getByText(
        "AutoPilot is yours alone — never shared. It listens, diagnoses, and builds the team that does the work.",
      ),
    ).toBeDefined();
    expect(screen.queryByText("Meet AutoPilot.")).toBeNull();

    const user = await advanceToCard(1);
    expect(
      await screen.findByText("Hire an expert, or raise your own."),
    ).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Next" }));
    // The connect card and its CTA survive the swap.
    expect(
      await screen.findByText("It works inside your tools."),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Connect your tools" }),
    ).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByText("Every morning, a briefing.")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.queryByText("It remembers everything.")).toBeNull();
  });

  it("keeps the old deck when only the team flag is on", async () => {
    flags.current = { "onboarding-expert-team": true };

    render(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);

    expect(await screen.findByText("Meet AutoPilot.")).toBeDefined();
  });
});

it("keeps the last card valid when team flags change in either direction", async () => {
  flags.current = { "onboarding-expert-team": true, "hire-experts": true };
  const { rerender } = render(
    <OnboardingWelcomeDialog isOpen onClose={vi.fn()} />,
  );
  await advanceToCard(3);
  expect(await screen.findByText("Every morning, a briefing.")).toBeDefined();
  flags.current = {};
  rerender(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);
  expect(await screen.findByText("It remembers everything.")).toBeDefined();
  flags.current = { "onboarding-expert-team": true, "hire-experts": true };
  rerender(<OnboardingWelcomeDialog isOpen onClose={vi.fn()} />);
  expect(await screen.findByText("Every morning, a briefing.")).toBeDefined();
});
