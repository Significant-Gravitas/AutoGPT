import { NextResponse, type NextRequest } from "next/server";
import { environment } from "@/services/environment";

export async function handleOnboardingRedirect(
  request: NextRequest,
  user: { id: string } | null,
  accessToken: string | null,
): Promise<NextResponse | null> {
  const pathname = request.nextUrl.pathname;
  const isOnboardingRoute = pathname.startsWith("/onboarding");
  const isResetRoute = pathname.startsWith("/onboarding/reset");

  // Skip if already on onboarding routes or reset route
  if (isOnboardingRoute || isResetRoute || !user || !accessToken) {
    return null;
  }

  try {
    // Check if onboarding is enabled
    const enabledResponse = await fetch(
      `${environment.getAGPTServerApiUrl()}/onboarding/enabled`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      },
    );

    if (!enabledResponse.ok) {
      return null;
    }

    const isEnabled = await enabledResponse.json();

    if (!isEnabled) {
      return null;
    }

    // Get user onboarding state
    const onboardingResponse = await fetch(
      `${environment.getAGPTServerApiUrl()}/onboarding`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      },
    );

    if (!onboardingResponse.ok) {
      return null;
    }

    const onboarding = await onboardingResponse.json();

    // If onboarding is complete, don't redirect
    if (
      onboarding.completedSteps.includes("GET_RESULTS") ||
      onboarding.completedSteps.includes("CONGRATS")
    ) {
      return null;
    }

    // User needs onboarding - redirect to /onboarding (client component will handle step redirects)
    const url = request.nextUrl.clone();
    url.pathname = "/onboarding";
    return NextResponse.redirect(url);
  } catch (error) {
    console.error("Failed to check onboarding status:", error);
    return null;
  }
}
