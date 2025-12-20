import { type NextRequest } from "next/server";
import { redirect } from "next/navigation";
import { environment } from "@/services/environment";

// Email confirmation route
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const token = searchParams.get("token");
  const next = searchParams.get("next") ?? "/";

  if (token) {
    try {
      const response = await fetch(
        `${environment.getAGPTServerBaseUrl()}/api/auth/verify-email?token=${encodeURIComponent(token)}`,
        {
          method: "GET",
        },
      );

      if (response.ok) {
        // redirect user to specified redirect URL or root of app
        redirect(next);
      }
    } catch (error) {
      console.error("Email verification error:", error);
    }
  }

  // redirect the user to an error page with some instructions
  redirect("/error");
}
