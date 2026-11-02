import {
  getPostAccountDeleteAccountMockHandler200,
  getPostAccountSendExitFeedbackMockHandler204,
} from "@/app/api/__generated__/endpoints/account/account.msw";
import {
  getGetSubscriptionStatusMockHandler200,
  getGetSubscriptionStatusResponseMock200,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { getGetV1ListCredentialsMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getGetV2ListMySkillSubmissionsMockHandler200,
  getGetV2ListMySkillSubmissionsResponseMock200,
  getGetV2ListMySubmissionsMockHandler200,
  getGetV2ListMySubmissionsResponseMock200,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { AccountDeletionRequestBody } from "@/app/api/__generated__/models/accountDeletionRequestBody";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { DeletionFeedbackRequest } from "@/app/api/__generated__/models/deletionFeedbackRequest";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";

const DELETION_PATH = "/api/proxy/api/account/deletion";
const CREDENTIAL_PATH =
  "/api/proxy/api/integrations/:provider/credentials/:credId";

/** Pins every field the flow reads; faker fills only the ones it never does. */
export function subscriptionStatus(
  overrides: Partial<SubscriptionStatusResponse> = {},
): SubscriptionStatusResponse {
  return getGetSubscriptionStatusResponseMock200({
    tier: "PRO",
    tier_costs: { PRO: 5000, MAX: 32000 },
    current_period_end: null,
    pending_tier: null,
    pending_tier_effective_at: null,
    pending_billing_cycle: null,
    ...overrides,
  });
}

function credential(
  overrides: Partial<CredentialsMetaResponse> &
    Pick<CredentialsMetaResponse, "id" | "provider">,
): CredentialsMetaResponse {
  return {
    type: "oauth2",
    title: null,
    scopes: [],
    username: null,
    is_managed: false,
    ...overrides,
  };
}

export const GMAIL = credential({
  id: "cred-gmail",
  provider: "google",
  service_name: "Gmail",
});

export const GMAIL_WORK = credential({
  id: "cred-gmail-work",
  provider: "google",
  service_name: "Gmail",
});

export const GOOGLE_CALENDAR = credential({
  id: "cred-calendar",
  provider: "google",
  service_name: "Google Calendar",
});

export const NOTION = credential({
  id: "cred-notion",
  provider: "notion",
  service_name: "Notion",
});

export const MANAGED_CREDENTIAL = credential({
  id: "cred-managed",
  provider: "openai",
  service_name: "OpenAI",
  is_managed: true,
});

export const SYSTEM_CREDENTIAL = credential({
  id: "cred-system",
  provider: "anthropic",
  type: "api_key",
  title: "Use credits for Anthropic",
  service_name: "Anthropic",
});

interface DeletionAPIArgs {
  subscription?: SubscriptionStatusResponse;
  credentials?: CredentialsMetaResponse[];
  hasListings?: boolean;
  hasSkillListings?: boolean;
}

/** The backend as the flow sees it before anyone clicks anything. */
export function mockAccountDeletionAPI({
  subscription = subscriptionStatus(),
  credentials = [],
  hasListings = false,
  hasSkillListings = false,
}: DeletionAPIArgs = {}) {
  server.use(
    getGetSubscriptionStatusMockHandler200(subscription),
    getGetV1ListCredentialsMockHandler200(credentials),
    getGetV2ListMySubmissionsMockHandler200({
      submissions: hasListings
        ? getGetV2ListMySubmissionsResponseMock200().submissions.slice(0, 1)
        : [],
      pagination: {
        total_items: hasListings ? 1 : 0,
        total_pages: hasListings ? 1 : 0,
        current_page: 1,
        page_size: 1,
      },
      stats: {
        total: 0,
        approved: 0,
        pending: 0,
        total_runs: 0,
        average_rating: null,
      },
    }),
    getGetV2ListMySkillSubmissionsMockHandler200(
      hasSkillListings
        ? getGetV2ListMySkillSubmissionsResponseMock200().slice(0, 1)
        : [],
    ),
    getPostAccountSendExitFeedbackMockHandler204(),
  );
}

export function recordFeedback(): DeletionFeedbackRequest[] {
  const bodies: DeletionFeedbackRequest[] = [];
  server.use(
    getPostAccountSendExitFeedbackMockHandler204(async ({ request }) => {
      bodies.push((await request.json()) as DeletionFeedbackRequest);
    }),
  );
  return bodies;
}

/** When a closed account's data may be erased, as the deletion route answers. */
export const ERASE_AFTER = new Date("2026-11-13T12:00:00Z");
export const CLOSED_TOAST =
  "Your account is closed. Sign in before Nov 13, 2026 if you change your mind.";

export function recordDeletion(): AccountDeletionRequestBody[] {
  const bodies: AccountDeletionRequestBody[] = [];
  server.use(
    getPostAccountDeleteAccountMockHandler200(async ({ request }) => {
      bodies.push((await request.json()) as AccountDeletionRequestBody);
      return { status: "scheduled", erase_after: ERASE_AFTER };
    }),
  );
  return bodies;
}

/** The generated 400 and 502 handlers send no body, but the real route sends
 *  its message as `detail`, which is what the flow shows. */
export function rejectDeletion(status: 400 | 502, detail: string) {
  const bodies: AccountDeletionRequestBody[] = [];
  server.use(
    http.post(DELETION_PATH, async ({ request }) => {
      bodies.push((await request.json()) as AccountDeletionRequestBody);
      return HttpResponse.json({ detail }, { status });
    }),
  );
  return bodies;
}

interface CredentialStoreArgs {
  failing?: string[];
  gate?: Promise<void>;
}

/** Connected apps that really go away when deleted, so a refetch after a
 *  partial failure lists only what is left. */
export function fakeCredentialStore(
  initial: CredentialsMetaResponse[],
  { failing = [], gate }: CredentialStoreArgs = {},
) {
  let stored = [...initial];
  const failingIDs = new Set(failing);
  const deleteURLs: string[] = [];

  server.use(
    getGetV1ListCredentialsMockHandler200(() => stored),
    http.delete(CREDENTIAL_PATH, async ({ request, params }) => {
      const url = new URL(request.url);
      deleteURLs.push(`${url.pathname}${url.search}`);
      await gate;
      if (failingIDs.has(String(params.credId))) {
        return HttpResponse.json(
          { detail: "Provider is unavailable" },
          { status: 500 },
        );
      }
      stored = stored.filter((item) => item.id !== params.credId);
      return HttpResponse.json({ deleted: true, revoked: true });
    }),
  );

  return {
    deleteURLs,
    stopFailing(id: string) {
      failingIDs.delete(id);
    },
  };
}

export function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
