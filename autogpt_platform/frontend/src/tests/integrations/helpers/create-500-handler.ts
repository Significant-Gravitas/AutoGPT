import { http, HttpResponse, delay } from "msw";

type HttpMethod = "get" | "post" | "put" | "patch" | "delete";

interface Create500HandlerOptions {
  delayMs?: number;
  body?: unknown;
}

export function create500Handler(
  method: HttpMethod,
  url: string,
  options?: Create500HandlerOptions,
) {
  const { delayMs = 0, body } = options ?? {};

  const responseBody = body ?? {
    detail: "Internal Server Error",
  };

  return http[method](url, async () => {
    if (delayMs > 0) {
      await delay(delayMs);
    }

    return HttpResponse.json(responseBody, {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  });
}
