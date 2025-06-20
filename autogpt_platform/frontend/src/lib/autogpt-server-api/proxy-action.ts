"use server";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import * as Sentry from "@sentry/nextjs";

export interface ProxyRequestOptions {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: string;
  payload?: Record<string, any>;
  baseUrl?: string;
}

export async function proxyApiRequest({
  method,
  path,
  payload,
  baseUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL ||
    "http://localhost:8006/api",
}: ProxyRequestOptions) {
  return await Sentry.withServerActionInstrumentation(
    "proxyApiRequest",
    {},
    async () => {
      const supabase = await getServerSupabase();

      if (!supabase) {
        throw new Error("Supabase client not available");
      }

      // Get the JWT token from server-side session
      let token = "no-token-found";
      try {
        const {
          data: { session },
          error,
        } = await supabase.auth.getSession();

        if (!error && session?.access_token) {
          token = session.access_token;
        }
      } catch (error) {
        console.error("Failed to get auth token:", error);
      }

      // Build the URL
      let url = baseUrl + path;
      const payloadAsQuery = ["GET", "DELETE"].includes(method);
      if (payloadAsQuery && payload) {
        const queryParams = new URLSearchParams(payload);
        url += `?${queryParams.toString()}`;
      }

      // Make the request
      const hasRequestBody = !payloadAsQuery && payload !== undefined;
      const response = await fetch(url, {
        method,
        headers: {
          ...(hasRequestBody && { "Content-Type": "application/json" }),
          ...(token &&
            token !== "no-token-found" && { Authorization: `Bearer ${token}` }),
        },
        body: hasRequestBody ? JSON.stringify(payload) : undefined,
      });

      if (!response.ok) {
        let errorDetail;
        try {
          const errorData = await response.json();
          if (
            Array.isArray(errorData.detail) &&
            errorData.detail.length > 0 &&
            errorData.detail[0].loc
          ) {
            // Pydantic validation error
            const errors = errorData.detail.map((err: any) => {
              const location = err.loc.join(" -> ");
              return `${location}: ${err.msg}`;
            });
            errorDetail = errors.join("\n");
          } else {
            errorDetail = errorData.detail || response.statusText;
          }
        } catch {
          errorDetail = response.statusText;
        }
        throw new Error(errorDetail);
      }

      // Handle responses with no content
      if (
        response.status === 204 ||
        response.headers.get("Content-Length") === "0"
      ) {
        return null;
      }

      try {
        return await response.json();
      } catch (e) {
        if (e instanceof SyntaxError) {
          return null;
        }
        throw e;
      }
    },
  );
}

export async function proxyFileUpload(
  path: string,
  formData: FormData,
  baseUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL ||
    "http://localhost:8006/api",
): Promise<string> {
  return await Sentry.withServerActionInstrumentation(
    "proxyFileUpload",
    {},
    async () => {
      const supabase = await getServerSupabase();

      if (!supabase) {
        throw new Error("Supabase client not available");
      }

      // Get the JWT token from server-side session
      let token = "no-token-found";
      try {
        const {
          data: { session },
          error,
        } = await supabase.auth.getSession();

        if (!error && session?.access_token) {
          token = session.access_token;
        }
      } catch (error) {
        console.error("Failed to get auth token:", error);
      }

      // Make the file upload request
      const response = await fetch(baseUrl + path, {
        method: "POST",
        headers: {
          ...(token &&
            token !== "no-token-found" && { Authorization: `Bearer ${token}` }),
          // Don't set Content-Type for FormData - let the browser set it with boundary
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error uploading file: ${response.statusText}`);
      }

      // Parse the response
      const media_url = await response.text();
      return media_url;
    },
  );
}
