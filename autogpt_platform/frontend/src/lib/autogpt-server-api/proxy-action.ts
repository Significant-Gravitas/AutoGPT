"use server";

import * as Sentry from "@sentry/nextjs";
import {
  buildRequestUrl,
  makeAuthenticatedFileUpload,
  makeAuthenticatedRequest,
} from "./helpers";

const DEFAULT_BASE_URL = "http://localhost:8006/api";

export interface ProxyRequestOptions {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: string;
  payload?: Record<string, any>;
  baseUrl?: string;
  contentType?: string;
}

export async function proxyApiRequest({
  method,
  path,
  payload,
  baseUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL || DEFAULT_BASE_URL,
  contentType = "application/json",
}: ProxyRequestOptions) {
  return await Sentry.withServerActionInstrumentation(
    "proxyApiRequest",
    {},
    async () => {
      const url = buildRequestUrl(baseUrl, path, method, payload);
      return makeAuthenticatedRequest(method, url, payload, contentType);
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
      const url = baseUrl + path;
      return makeAuthenticatedFileUpload(url, formData);
    },
  );
}
