"use server";

import * as Sentry from "@sentry/nextjs";
import {
  buildRequestUrl,
  makeAuthenticatedFileUpload,
  makeAuthenticatedRequest,
} from "./helpers";
import { getAgptServerApiUrl } from "@/lib/env-config";

export interface ProxyRequestOptions {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: string;
  payload?: Record<string, any>;
  contentType?: string;
}

export async function proxyApiRequest({
  method,
  path,
  payload,
  contentType = "application/json",
}: ProxyRequestOptions) {
  return await Sentry.withServerActionInstrumentation(
    "proxyApiRequest",
    {},
    async () => {
      const baseUrl = getAgptServerApiUrl();
      const url = buildRequestUrl(baseUrl, path, method, payload);
      return makeAuthenticatedRequest(method, url, payload, contentType);
    },
  );
}

export async function proxyFileUpload(
  path: string,
  formData: FormData,
): Promise<string> {
  return await Sentry.withServerActionInstrumentation(
    "proxyFileUpload",
    {},
    async () => {
      const baseUrl = getAgptServerApiUrl();
      const url = baseUrl + path;
      return makeAuthenticatedFileUpload(url, formData);
    },
  );
}
