import { auth } from "@/lib/auth/auth";
import { isAPIError } from "better-auth/api";
import { NextResponse } from "next/server";
import { z } from "zod";

const MAX_ACCOUNTS = 500;

// Bans run one at a time, a few queries each: the largest batch has to fit.
export const maxDuration = 300;

const bodySchema = z.object({
  userIds: z.array(z.string().min(1)).min(1).max(MAX_ACCOUNTS),
  reason: z.string().trim().min(1).max(500),
  banExpiresIn: z.number().int().positive().optional(),
});

interface BanAccountResult {
  userId: string;
  ok: boolean;
  error?: string;
}

// Bans exactly the listed accounts through Better Auth's admin banUser, one by
// one, and answers per account, so a failure halfway leaves no guessing.
export async function POST(request: Request) {
  const session = await auth.api.getSession({ headers: request.headers });
  if (!session?.user) {
    return NextResponse.json({ error: "Not signed in" }, { status: 401 });
  }
  if (session.user.role !== "admin") {
    return NextResponse.json(
      { error: "Admin access required" },
      { status: 403 },
    );
  }

  const parsed = bodySchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }
  const { userIds, reason, banExpiresIn } = parsed.data;

  const results: BanAccountResult[] = [];
  for (const userId of new Set(userIds)) {
    results.push(await banAccount({ request, userId, reason, banExpiresIn }));
  }
  return NextResponse.json({ results });
}

interface BanArgs {
  request: Request;
  userId: string;
  reason: string;
  banExpiresIn: number | undefined;
}

async function banAccount({
  request,
  userId,
  reason,
  banExpiresIn,
}: BanArgs): Promise<BanAccountResult> {
  try {
    await auth.api.banUser({
      headers: request.headers,
      body: { userId, banReason: reason, banExpiresIn },
    });
    return { userId, ok: true };
  } catch (error) {
    return {
      userId,
      ok: false,
      error: isAPIError(error)
        ? error.body?.message || error.message
        : "Failed to ban the account",
    };
  }
}
