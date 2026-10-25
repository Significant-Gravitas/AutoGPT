import { getUpdateDelegationSettingsMockHandler422 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";
import type { DelegationSettings } from "@/app/api/__generated__/models/delegationSettings";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import AutopilotPage from "../page";
import { mockOttoApi, SETTINGS } from "./delegationFixtures";

// The tabs live in ?tab=; a remembering adapter keeps a clicked tab open.
function renderPage() {
  return render(
    <NuqsTestingAdapter hasMemory>
      <AutopilotPage />
    </NuqsTestingAdapter>,
  );
}

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
    renderPage();
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
    renderPage();
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
    renderPage();
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
    renderPage();
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

  test("two quick changes both reach the server", async () => {
    mockOttoApi();
    const server_ = holdFirstSave();
    renderPage();
    await openSettings();

    const toggle = await screen.findByRole("switch", {
      name: "New experts start in ask-first",
    });
    fireEvent.click(toggle);
    await waitFor(() => expect(server_.received).toHaveLength(1));
    // Changed while the first save is still on its way.
    fireEvent.click(
      screen.getByRole("combobox", { name: "Per-delegation cap" }),
    );
    fireEvent.click(await screen.findByRole("option", { name: "$5.00" }));
    await waitFor(() => expect(server_.settled()).toBe(true));

    await waitFor(() =>
      expect(server_.current()).toEqual({
        ...SETTINGS,
        new_experts_ask_first: true,
        per_delegation_cap_usd: 5,
      }),
    );
    expect(toggle.getAttribute("aria-checked")).toBe("true");
    expect(
      screen.getByRole("combobox", { name: "Per-delegation cap" }).textContent,
    ).toBe("$5.00");
  });

  test("a change that fails is undone alone, and the next one still saves", async () => {
    mockOttoApi();
    const server_ = holdFirstSave({ fail: true });
    renderPage();
    await openSettings();

    const toggle = await screen.findByRole("switch", {
      name: "New experts start in ask-first",
    });
    fireEvent.click(toggle);
    await waitFor(() => expect(server_.received).toHaveLength(1));
    fireEvent.click(
      screen.getByRole("combobox", { name: "Per-delegation cap" }),
    );
    fireEvent.click(await screen.findByRole("option", { name: "$5.00" }));
    await waitFor(() => expect(server_.settled()).toBe(true));

    await waitFor(() =>
      expect(server_.current()).toEqual({
        ...SETTINGS,
        per_delegation_cap_usd: 5,
      }),
    );
    await waitFor(() =>
      expect(toggle.getAttribute("aria-checked")).toBe("false"),
    );
    expect(
      screen.getByRole("combobox", { name: "Per-delegation cap" }).textContent,
    ).toBe("$5.00");
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't save the delegation settings",
      }),
    );
  });
});

/** A settings server that answers the first save last when the page sends
 *  a second save before the first is answered (it holds the first until the
 *  second arrives), or after a short wait when it doesn't. With ``fail`` the
 *  first save is refused. */
function holdFirstSave({ fail = false } = {}) {
  let current: DelegationSettings = { ...SETTINGS };
  const received: DelegationSettings[] = [];
  let answered = 0;
  let secondArrived = () => {};
  const second = new Promise<void>((resolve) => {
    secondArrived = resolve;
  });
  server.use(
    http.get("/api/proxy/api/experts/delegation-settings", () =>
      HttpResponse.json(current),
    ),
    http.put(
      "/api/proxy/api/experts/delegation-settings",
      async ({ request }) => {
        const body = (await request.json()) as DelegationSettings;
        received.push(body);
        if (received.length === 2) secondArrived();
        if (received.length === 1) {
          await Promise.race([
            second.then(() => new Promise((r) => setTimeout(r, 50))),
            new Promise((r) => setTimeout(r, 500)),
          ]);
          if (fail) {
            answered += 1;
            return HttpResponse.json({ detail: "Not saved" }, { status: 422 });
          }
        }
        current = body;
        answered += 1;
        return HttpResponse.json(current);
      },
    ),
  );
  return {
    received,
    current: () => current,
    // Both saves have their answer.
    settled: () => received.length >= 2 && answered === received.length,
  };
}
