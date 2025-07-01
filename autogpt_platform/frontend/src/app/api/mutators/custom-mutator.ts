const BASE_URL = `${process.env.NEXT_PUBLIC_FRONTEND_BASE_URL}/api/proxy`; // Sending request via nextjs Server

console.log("🔧 Custom Mutator - Base URL:", BASE_URL);
console.log("🔧 Environment variables:", {
  NEXT_PUBLIC_FRONTEND_BASE_URL: process.env.NEXT_PUBLIC_FRONTEND_BASE_URL,
  NODE_ENV: process.env.NODE_ENV,
});

const getBody = <T>(c: Response | Request): Promise<T> => {
  const contentType = c.headers.get("content-type");

  console.log("📦 Parsing response body:", {
    contentType,
    headers: Object.fromEntries(c.headers.entries()),
  });

  if (contentType && contentType.includes("application/json")) {
    console.log("📄 Parsing as JSON");
    return c.json();
  }

  if (contentType && contentType.includes("application/pdf")) {
    console.log("📄 Parsing as PDF blob");
    return c.blob() as Promise<T>;
  }

  console.log("📄 Parsing as text");
  return c.text() as Promise<T>;
};

export const customMutator = async <T = any>(
  url: string,
  options: RequestInit & {
    params?: any;
  } = {},
): Promise<T> => {
  console.log("🚀 === CUSTOM MUTATOR START ===");

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

  console.log("📋 Request preparation:", {
    url,
    method,
    params,
    hasBody: !!data,
    bodyType: data instanceof FormData ? "FormData" : typeof data,
    originalHeaders: requestOptions.headers,
  });

  const isFormData = data instanceof FormData;

  // Currently, only two content types are handled here: application/json and multipart/form-data
  if (!isFormData && data && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const queryString = params
    ? "?" + new URLSearchParams(params).toString()
    : "";

  const finalUrl = `${BASE_URL}${url}${queryString}`;

  console.log("🎯 Final request details:", {
    finalUrl,
    method,
    headers,
    queryString,
    bodyPreview: isFormData
      ? "FormData (entries hidden)"
      : data
        ? String(data).substring(0, 200) +
          (String(data).length > 200 ? "..." : "")
        : "No body",
  });

  try {
    console.log("📡 Making fetch request...");
    const response = await fetch(finalUrl, {
      ...requestOptions,
      method,
      headers,
      body: data,
    });

    console.log("📨 Response received:", {
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      headers: Object.fromEntries(response.headers.entries()),
      url: response.url,
    });

    const response_data = await getBody<T>(response);

    console.log("✅ Response data parsed:", {
      dataType: typeof response_data,
      dataPreview:
        typeof response_data === "object"
          ? JSON.stringify(response_data).substring(0, 200) + "..."
          : String(response_data).substring(0, 200),
    });

    const result = {
      status: response.status,
      data: response_data,
      headers: response.headers,
    } as T;

    console.log("🏁 === CUSTOM MUTATOR END ===");
    return result;
  } catch (error) {
    console.error("💥 === CUSTOM MUTATOR FAILED ===");
    console.error("❌ Fetch error:", error);
    console.error("❌ Error details:", {
      name: error instanceof Error ? error.name : "Unknown",
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : "No stack trace",
      cause: error instanceof Error ? error.cause : undefined,
    });
    console.error("❌ Request context:", {
      finalUrl,
      method,
      headers,
    });

    throw error;
  }
};
