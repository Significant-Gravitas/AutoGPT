import { getSupabaseClient } from "@/lib/supabase/getSupabaseClient";

const BASE_URL =
  process.env.NEXT_PUBLIC_AGPT_SERVER_BASE_URL || "http://localhost:8006";

const getBody = <T>(c: Response | Request): Promise<T> => {
  const contentType = c.headers.get("content-type");

  if (contentType && contentType.includes("application/json")) {
    return c.json();
  }

  if (contentType && contentType.includes("application/pdf")) {
    return c.blob() as Promise<T>;
  }

  return c.text() as Promise<T>;
};

const getSupabaseToken = async () => {
  const supabase = await getSupabaseClient();

  const {
    data: { session },
  } = (await supabase?.auth.getSession()) || {
    data: { session: null },
  };

  return session?.access_token;
};

export const customMutator = async <T = any>(
  url: string,
  options: RequestInit & {
    params?: any;
  } = {},
): Promise<T> => {
  const { params, ...requestOptions } = options;
  const method = (requestOptions.method || "GET") as
    | "GET"
    | "POST"
    | "PUT"
    | "DELETE"
    | "PATCH";
  const data = requestOptions.body;
  const headers: Record<string, string> = {
    ...((requestOptions.headers as Record<string, string>) || {}),
  };

  const token = await getSupabaseToken();

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const isFormData = data instanceof FormData;

  // Currently, only two content types are handled here: application/json and multipart/form-data
  if (!isFormData && data && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const queryString = params
    ? "?" + new URLSearchParams(params).toString()
    : "";

  const response = await fetch(`${BASE_URL}${url}${queryString}`, {
    ...requestOptions,
    method,
    headers,
    body: data,
  });

  const response_data = await getBody<T>(response);

  return {
    status: response.status,
    response_data,
    headers: response.headers,
  } as T;
};
