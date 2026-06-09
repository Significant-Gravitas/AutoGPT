import { getOnboardingStatus } from "@/app/api/helpers";
import BackendAPI from "@/lib/autogpt-server-api";
import { getServerAuthSession } from "@/lib/auth/auth";
import { revalidatePath } from "next/cache";
import { NextResponse } from "next/server";

// Handle the callback to complete the user session login
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  let next = "/copilot";
  const session = await getServerAuthSession();

  if (!session) {
    return NextResponse.redirect(`${origin}/auth/auth-code-error`);
  }

  try {
    const api = new BackendAPI();
    await api.createUser();

    const { shouldShowOnboarding } = await getOnboardingStatus();
    next = shouldShowOnboarding ? "/onboarding" : "/copilot";
    revalidatePath(next, "layout");
  } catch (createUserError) {
    console.error("Error creating user:", createUserError);

    if (
      createUserError &&
      typeof createUserError === "object" &&
      "status" in createUserError
    ) {
      const apiError = createUserError as { status?: number };

      if (apiError.status === 401) {
        return NextResponse.redirect(
          `${origin}/error?message=auth-token-invalid`,
        );
      }
      if (typeof apiError.status === "number" && apiError.status >= 500) {
        return NextResponse.redirect(`${origin}/error?message=server-error`);
      }
      if (apiError.status === 429) {
        return NextResponse.redirect(`${origin}/error?message=rate-limited`);
      }
    }

    if (
      createUserError instanceof TypeError &&
      createUserError.message.includes("fetch")
    ) {
      return NextResponse.redirect(`${origin}/error?message=network-error`);
    }

    return NextResponse.redirect(
      `${origin}/error?message=user-creation-failed`,
    );
  }

  next = searchParams.get("next") || next;

  const forwardedHost = request.headers.get("x-forwarded-host");
  const isLocalEnv = process.env.NODE_ENV === "development";

  if (isLocalEnv) {
    return NextResponse.redirect(`${origin}${next}`);
  }

  if (forwardedHost) {
    return NextResponse.redirect(`https://${forwardedHost}${next}`);
  }

  return NextResponse.redirect(`${origin}${next}`);
}
