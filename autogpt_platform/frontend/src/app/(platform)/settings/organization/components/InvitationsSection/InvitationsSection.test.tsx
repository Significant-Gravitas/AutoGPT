import {
  getGetV2ListPendingInvitationsMockHandler,
  getPostV2ResendInvitationMockHandler,
} from "@/app/api/__generated__/endpoints/invitations/invitations.msw";
import { getGetV2ListWorkspacesMockHandler } from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import type { InvitationCreateResponse } from "@/app/api/__generated__/models/invitationCreateResponse";
import type { InvitationResponse } from "@/app/api/__generated__/models/invitationResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { InvitationsSection } from "./InvitationsSection";

const PENDING_INVITE: InvitationResponse = {
  id: "inv-1",
  email: "pending@acme.test",
  is_admin: false,
  is_billing_manager: false,
  expires_at: new Date("2026-08-01T00:00:00Z"),
  created_at: new Date("2026-07-01T00:00:00Z"),
  team_ids: [],
};

const RESEND_RESPONSE: InvitationCreateResponse = {
  ...PENDING_INVITE,
  token: "tok-new",
};

describe("InvitationsSection", () => {
  it("resends a pending invitation and refreshes the list", async () => {
    let listCalls = 0;
    let resendUrl = "";
    const resendSpy = vi.fn();

    server.use(
      getGetV2ListWorkspacesMockHandler([]),
      getGetV2ListPendingInvitationsMockHandler(() => {
        listCalls += 1;
        return [PENDING_INVITE];
      }),
      getPostV2ResendInvitationMockHandler((info) => {
        resendSpy();
        resendUrl = info.request.url;
        return RESEND_RESPONSE;
      }),
    );

    render(<InvitationsSection orgId="org-1" isAdmin />);

    await screen.findByText("pending@acme.test");
    const listCallsBeforeResend = listCalls;

    await userEvent.click(screen.getByRole("button", { name: "Resend" }));

    await waitFor(() => expect(resendSpy).toHaveBeenCalledTimes(1));
    expect(resendUrl).toContain("/api/orgs/org-1/invitations/inv-1/resend");
    await waitFor(() =>
      expect(listCalls).toBeGreaterThan(listCallsBeforeResend),
    );
  });

  it("renders nothing for non-admins", () => {
    render(<InvitationsSection orgId="org-1" isAdmin={false} />);

    expect(screen.queryByTestId("org-invitations-section")).toBeNull();
  });
});
