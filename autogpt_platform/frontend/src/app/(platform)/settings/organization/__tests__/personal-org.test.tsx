import {
  getGetV2ListUserOrganizationsMockHandler,
  getPostV2ConvertPersonalOrgToTeamOrgMockHandler,
  getPostV2ConvertPersonalOrgToTeamOrgMockHandler422,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import OrganizationSettingsPage from "../page";
import {
  OWNER_MEMBER,
  PENDING_INVITATION_FROM_OTHER_ORG,
  PERSONAL_ORG,
  mockOrg,
  mockOrgDetails,
  seedActiveOrg,
} from "./fixtures";

const toastSpy = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-owner" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

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

const CONVERTED_ORG: OrgResponse = { ...PERSONAL_ORG, is_personal: false };

const NEW_PERSONAL_ORG: OrgResponse = {
  ...PERSONAL_ORG,
  id: "org-personal-new",
  slug: "jane-personal-1",
};

function mockPersonalOrg(canCreateOrganization = true) {
  mockOrg({
    org: PERSONAL_ORG,
    members: [OWNER_MEMBER],
    canCreateOrganization,
  });
}

async function openConvertDialog() {
  const section = await screen.findByTestId("org-convert-section");
  await userEvent.click(
    within(section).getByRole("button", {
      name: "Convert to a team organization",
    }),
  );
  return screen.findByRole("dialog");
}

afterEach(() => {
  delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
});

describe("Personal organization settings", () => {
  beforeEach(() => {
    window.localStorage.clear();
    toastSpy.mockClear();
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true";
    seedActiveOrg(PERSONAL_ORG);
  });

  it("shows only General settings, with no members or invitations", async () => {
    mockPersonalOrg();
    render(<OrganizationSettingsPage />);

    expect(await screen.findByTestId("org-profile-section")).toBeDefined();
    expect(screen.queryByRole("tablist")).toBeNull();
    expect(screen.queryByTestId("org-members-section")).toBeNull();
    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
    expect(screen.queryByTestId("org-team-spend-section")).toBeNull();
    expect(screen.queryByText("Members")).toBeNull();
  });

  it("offers to create a team organization", async () => {
    mockPersonalOrg();
    render(<OrganizationSettingsPage />);

    const prompt = await screen.findByTestId("create-team-org-prompt");
    expect(within(prompt).getByText("Want to work with others?")).toBeDefined();
    await userEvent.click(
      within(prompt).getByRole("button", {
        name: "Create a team organization",
      }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByPlaceholderText("Acme Inc.")).toBeDefined();
    expect(
      within(dialog).getByRole("button", { name: "Create organization" }),
    ).toBeDefined();
  });

  it("hides creating and converting from an account that can't create organizations", async () => {
    mockPersonalOrg(false);
    render(<OrganizationSettingsPage />);

    await screen.findByTestId("org-profile-section");
    await waitFor(() => {
      expect(screen.queryByTestId("create-team-org-prompt")).toBeNull();
    });
    expect(screen.queryByTestId("org-convert-section")).toBeNull();
  });

  it("still shows pending invitations, so an invitee can accept them", async () => {
    mockOrg({
      org: PERSONAL_ORG,
      members: [OWNER_MEMBER],
      myInvitations: [PENDING_INVITATION_FROM_OTHER_ORG],
    });
    render(<OrganizationSettingsPage />);

    const invitations = await screen.findByTestId("my-invitations-section");
    expect(within(invitations).getByText("Other Corp")).toBeDefined();
    expect(
      within(invitations).getByRole("button", { name: "Accept" }),
    ).toBeDefined();
  });

  it("says what converting does before it happens", async () => {
    mockPersonalOrg();
    render(<OrganizationSettingsPage />);

    const dialog = await openConvertDialog();
    expect(
      within(dialog).getByText(
        "Your workflows and store listings move into the new team organization.",
      ),
    ).toBeDefined();
    expect(
      within(dialog).getByText("You get a fresh personal organization."),
    ).toBeDefined();
    expect(
      within(dialog).getByText(
        "The team organization starts with an empty wallet. It needs a team plan, set by an AutoGPT admin, before people can use it.",
      ),
    ).toBeDefined();
  });

  it("converts the org, refreshes the org list and keeps the user in the converted org", async () => {
    let convertedPath: string | null = null;
    let isConverted = false;
    mockPersonalOrg();
    mockOrgDetails(() => (isConverted ? CONVERTED_ORG : PERSONAL_ORG));
    server.use(
      getPostV2ConvertPersonalOrgToTeamOrgMockHandler((info) => {
        convertedPath = new URL(info.request.url).pathname;
        isConverted = true;
        return CONVERTED_ORG;
      }),
      getGetV2ListUserOrganizationsMockHandler([
        CONVERTED_ORG,
        NEW_PERSONAL_ORG,
      ]),
    );
    render(<OrganizationSettingsPage />);

    const dialog = await openConvertDialog();
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Convert organization" }),
    );

    await waitFor(() => {
      expect(convertedPath).toBe(
        `/api/proxy/api/orgs/${PERSONAL_ORG.id}/convert`,
      );
    });
    await waitFor(() => {
      const orgs = useOrgTeamStore.getState().orgs;
      expect(orgs.map((org) => [org.id, org.isPersonal])).toEqual([
        [PERSONAL_ORG.id, false],
        [NEW_PERSONAL_ORG.id, true],
      ]);
    });
    const state = useOrgTeamStore.getState();
    expect(state.activeOrgID).toBe(PERSONAL_ORG.id);
    expect(state.isLoaded).toBe(true);

    // The page now shows the converted org's team settings.
    expect(await screen.findByRole("tab", { name: "Members" })).toBeDefined();
    expect(screen.queryByTestId("org-convert-section")).toBeNull();
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "success" }),
    );
  });

  it("shows a failure toast and stays personal when converting fails", async () => {
    mockPersonalOrg();
    server.use(getPostV2ConvertPersonalOrgToTeamOrgMockHandler422());
    render(<OrganizationSettingsPage />);

    const dialog = await openConvertDialog();
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Convert organization" }),
    );

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't convert this organization",
          variant: "destructive",
        }),
      );
    });
    expect(useOrgTeamStore.getState().orgs[0].isPersonal).toBe(true);
    expect(screen.getByTestId("org-convert-section")).toBeDefined();
  });
});
