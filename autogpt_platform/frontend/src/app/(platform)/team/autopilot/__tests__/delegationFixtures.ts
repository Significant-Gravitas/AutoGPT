import {
  getGetDelegationSettingsMockHandler200,
  getListDelegationsMockHandler200,
  getListExpertsMockHandler,
  getUpdateDelegationSettingsMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import type { DelegationSettings } from "@/app/api/__generated__/models/delegationSettings";
import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import type { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import type { DelegationCounts } from "@/app/api/__generated__/models/delegationCounts";
import { server } from "@/mocks/mock-server";
import {
  type DelegationFilter,
  filterDelegations,
} from "../../components/DelegationList/helpers";

const NOW = new Date();
const YESTERDAY = new Date(NOW.getTime() - 26 * 60 * 60 * 1000);
const DEVON = { id: "exp-devon", name: "Devon", role: "Engineer" };
const ALEX = { id: "exp-alex", name: "Alex", role: "Product Manager" };

export function delegation(
  over: Partial<DelegationSummary>,
): DelegationSummary {
  return {
    sub_session_id: "sub-1",
    review_id: null,
    parent_session_id: "otto-chat",
    expert: DEVON,
    title: "Eng estimate for onboarding revamp",
    brief: "Estimating from the acceptance criteria.",
    status: "running",
    created_at: NOW,
    finished_at: null,
    elapsed_seconds: 60,
    cost_usd: 0.04,
    files_count: 0,
    question: null,
    question_options: [],
    asked_at: null,
    delegated_by_expert_id: null,
    ...over,
  };
}

export const DELEGATIONS = [
  delegation({
    sub_session_id: null,
    review_id: "rev-1",
    title: "Retention policy check",
    brief: "Devon wants to run the check before estimating.",
    status: "proposed",
    cost_usd: null,
  }),
  delegation({}),
  delegation({
    sub_session_id: "sub-2",
    expert: ALEX,
    title: "Onboarding revamp PRD",
    brief: "PRD draft with 9 acceptance criteria.",
    status: "completed",
    elapsed_seconds: 400,
    cost_usd: 0.31,
    files_count: 1,
  }),
  delegation({
    sub_session_id: "sub-3",
    title: "Partner outreach list",
    brief: "Stopped at the weekly budget cap.",
    status: "failed",
    created_at: YESTERDAY,
  }),
];

export const SETTINGS: DelegationSettings = {
  mode: "ask_first",
  per_delegation_cap_usd: 2,
  daily_budget_usd: 10,
  ask_before_external: true,
  ask_before_over_cap: true,
  new_experts_ask_first: false,
};

/** An empty team, Otto's hand-offs, and settings the server keeps. */
export function mockOttoApi() {
  let current: DelegationSettings = { ...SETTINGS };
  const saved: DelegationSettings[] = [];
  server.use(
    getListExpertsMockHandler([]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200({
      agents: [],
      pagination: {
        total_items: 0,
        total_pages: 1,
        current_page: 1,
        page_size: 100,
      },
    } as LibraryAgentResponse),
    getListCopilotSkillsMockHandler200([]),
    listDelegationsLikeTheServer(DELEGATIONS, {
      working: 1,
      needs_you: 1,
      completed: 1,
      failed: 1,
      started_today: 3,
      spent_today_usd: 0.4,
    }),
    getGetDelegationSettingsMockHandler200(() => current),
    getUpdateDelegationSettingsMockHandler200(async ({ request }) => {
      current = (await request.json()) as DelegationSettings;
      saved.push(current);
      return current;
    }),
  );
  return { saved };
}

/** The list endpoint as the server answers it: filtered by ``group`` and
 *  cut to ``offset``/``limit``, with counts over every hand-off. */
export function listDelegationsLikeTheServer(
  rows: DelegationSummary[],
  summary: DelegationCounts,
  onRequest?: () => void,
) {
  return getListDelegationsMockHandler200(({ request }) => {
    onRequest?.();
    const params = new URL(request.url).searchParams;
    const group = (params.get("group") ?? "all") as DelegationFilter;
    const offset = Number(params.get("offset") ?? 0);
    const limit = Number(params.get("limit") ?? 50);
    const matching = filterDelegations(rows, group);
    return {
      delegations: matching.slice(offset, offset + limit),
      total: matching.length,
      summary,
    };
  });
}

/** No hand-offs: registered ahead of `/experts/:expertId`, which would
 *  otherwise answer `/experts/delegations` with an expert. */
export function noDelegations() {
  return listDelegationsLikeTheServer([], {
    working: 0,
    needs_you: 0,
    completed: 0,
    failed: 0,
    started_today: 0,
    spent_today_usd: 0,
  });
}
