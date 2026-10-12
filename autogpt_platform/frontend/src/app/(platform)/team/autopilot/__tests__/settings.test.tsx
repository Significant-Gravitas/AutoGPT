import { getUpdateDelegationSettingsMockHandler422 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import AutopilotPage from "../page";
import { mockOttoApi, SETTINGS } from "./delegationFixtures";

const { toastSpy } = vi.hoisted(() => ({ toastSpy: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/team/autopilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

async function openSettings() {
  await userEvent.click(await screen.findByRole("tab", { name: "Settings" }));
}

describe("Otto's Settings tab", () => {
  test("shows the saved mode, caps and toggles", async () => {
    mockOttoApi();
    render(<AutopilotPage />);
    await openSettings();

    expect(
      (await screen.findByRole("combobox", { name: "Delegation mode" }))
        .textContent,
    ).toBe("Ask first");
    expect(
      screen.getByRole("combobox", { name: "Per-delegation cap" }).textContent,
    ).toBe("$2.00");
    expect(
      screen.getByRole("combobox", { name: "Daily delegation budget" })
        .textContent,
    ).toBe("$10.00");
    expect(
      screen
        .getByRole("switch", {
          name: "Ask before sending anything outside the workspace",
        })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  test("saves the whole settings object with the changed cap", async () => {
    const { saved } = mockOttoApi();
    render(<AutopilotPage />);
    await openSettings();

    fireEvent.click(
      await screen.findByRole("combobox", { name: "Per-delegation cap" }),
    );
    fireEvent.click(await screen.findByRole("option", { name: "$5.00" }));

    await waitFor(() => expect(saved).toHaveLength(1));
    expect(saved[0]).toEqual({ ...SETTINGS, per_delegation_cap_usd: 5 });
  });

  test("flips a toggle and saves it", async () => {
    const { saved } = mockOttoApi();
    render(<AutopilotPage />);
    await openSettings();

    const toggle = await screen.findByRole("switch", {
      name: "New experts start in ask-first",
    });
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(toggle);

    await waitFor(() => expect(saved).toHaveLength(1));
    expect(saved[0]).toEqual({ ...SETTINGS, new_experts_ask_first: true });
    await waitFor(() =>
      expect(toggle.getAttribute("aria-checked")).toBe("true"),
    );
  });

  test("puts the toggle back and says so when the save fails", async () => {
    mockOttoApi();
    server.use(getUpdateDelegationSettingsMockHandler422());
    render(<AutopilotPage />);
    await openSettings();

    const toggle = await screen.findByRole("switch", {
      name: "Ask before going over the cap",
    });
    fireEvent.click(toggle);

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't save the delegation settings",
          variant: "destructive",
        }),
      ),
    );
    await waitFor(() =>
      expect(toggle.getAttribute("aria-checked")).toBe("true"),
    );
  });
});
