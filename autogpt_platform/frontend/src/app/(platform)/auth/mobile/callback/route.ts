import { mobileCallbackPage } from "@/lib/auth/mobile-callback-page";

export const dynamic = "force-dynamic";

export function GET() {
  return mobileCallbackPage();
}
