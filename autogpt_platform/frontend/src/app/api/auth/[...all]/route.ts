import { auth } from "@/lib/auth/auth";
import { toNextJsHandler } from "better-auth/next-js";

const handlers = toNextJsHandler(auth);

export const POST = handlers.POST;

export async function GET(request: Request) {
  const response = await handlers.GET(request);
  if (new URL(request.url).pathname !== "/api/auth/token") return response;
  // The backend JWT must never be stored by a browser or shared cache.
  const uncached = new Response(response.body, response);
  uncached.headers.set("Cache-Control", "private, no-store");
  return uncached;
}
