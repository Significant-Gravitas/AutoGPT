import { auth } from "@/lib/auth/auth";
import { holdVerifyLink } from "@/lib/auth/verify-email-confirm";
import { toNextJsHandler } from "better-auth/next-js";

const handlers = toNextJsHandler(auth);

export const POST = handlers.POST;

export async function GET(request: Request) {
  return holdVerifyLink(request) ?? handlers.GET(request);
}
