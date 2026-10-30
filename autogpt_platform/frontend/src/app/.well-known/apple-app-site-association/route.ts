import { appleAppSiteAssociationResponse } from "@/lib/auth/mobile-app-links";

export const dynamic = "force-dynamic";

export function GET() {
  return appleAppSiteAssociationResponse();
}
