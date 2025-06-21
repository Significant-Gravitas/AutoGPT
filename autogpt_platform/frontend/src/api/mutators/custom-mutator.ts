import { getSupabaseClient } from "@/lib/supabase/getSupabaseClient";

export const customMutator = async ({
  url,
  method,
  params,
  data,
}: {
  url: string;
  method: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  params?: any;
  data?: FormData | any;
}) => {
  const headers: Record<string, string> = {};
  const supabase = await getSupabaseClient();

  const {
    data: { session },
  } = (await supabase?.auth.getSession()) || {
    data: { session: null },
  };

  if (session?.access_token) {
    headers["Authorization"] = `Bearer ${session.access_token}`;
  }

  const isFormData = data instanceof FormData;

  if (!isFormData && data) {
    headers["Content-Type"] = "application/json";
  }

  const queryString = params
    ? "?" + new URLSearchParams(params).toString()
    : "";

  const response = await fetch(`${url}${queryString}`, {
    method,
    headers,
    body: isFormData ? data : data ? JSON.stringify(data) : undefined,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
};

export type BodyType = FormData | any;
