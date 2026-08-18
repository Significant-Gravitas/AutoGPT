import * as Sentry from "@sentry/nextjs";
import { cookies } from "next/headers";
import { after } from "next/server";
import {
  ANALYTICS_CONSENT_COOKIE,
  ANALYTICS_CONSENT_GRANTED,
} from "@/services/consent/constants";
import { environment } from "@/services/environment";

const DATAFAST_GOALS_URL = "https://datafa.st/api/v1/goals";
const DATAFAST_VISITOR_COOKIE = "datafast_visitor_id";
const USER_CREATED_HEADER = "X-AutoGPT-User-Created";
const VISITOR_ID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

type SignupMethod = "email" | "google";

interface GoalContext {
  apiKey: string;
  method: SignupMethod;
  visitorID: string;
}

interface AccountCreationResponse {
  headers: Headers;
  status: number;
}

export function wasAccountCreated(response: AccountCreationResponse) {
  if (response.status !== 200) {
    throw Object.assign(
      new Error(`Unexpected account creation status ${response.status}`),
      { status: response.status },
    );
  }

  return response.headers.get(USER_CREATED_HEADER) === "true";
}

export async function scheduleAccountCreatedGoal(method: SignupMethod) {
  try {
    const context = await getGoalContext(method);
    if (!context) return;

    after(async () => {
      try {
        await sendAccountCreatedGoal(context);
      } catch (error) {
        reportTrackingError(error);
      }
    });
  } catch (error) {
    reportTrackingError(error);
  }
}

function reportTrackingError(error: unknown) {
  try {
    Sentry.captureException(error, {
      tags: { analytics_provider: "datafast", analytics_goal: "signup" },
    });
  } catch {
    console.error("Failed to report DataFast signup goal error");
  }
}

async function getGoalContext(
  method: SignupMethod,
): Promise<GoalContext | null> {
  const cookieStore = await cookies();
  const hasConsent =
    cookieStore.get(ANALYTICS_CONSENT_COOKIE)?.value ===
    ANALYTICS_CONSENT_GRANTED;
  if (!hasConsent) return null;

  const visitorID = cookieStore.get(DATAFAST_VISITOR_COOKIE)?.value;
  if (!visitorID || !VISITOR_ID_PATTERN.test(visitorID)) return null;

  const apiKey = process.env.DATAFAST_API_KEY?.trim();
  if (!apiKey) {
    if (!environment.isCloud()) return null;

    throw new Error("DATAFAST_API_KEY must be configured in cloud");
  }

  if (!apiKey.startsWith("df_")) {
    throw new Error("DATAFAST_API_KEY must use a df_ website key");
  }

  return { apiKey, method, visitorID };
}

async function sendAccountCreatedGoal(context: GoalContext) {
  const response = await fetch(DATAFAST_GOALS_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${context.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      datafast_visitor_id: context.visitorID,
      name: "signup",
      metadata: { method: context.method },
    }),
    cache: "no-store",
    signal: AbortSignal.timeout(5_000),
  });

  if (!response.ok) {
    throw new Error(
      `DataFast signup goal failed with status ${response.status}`,
    );
  }
}
