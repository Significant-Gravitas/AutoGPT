"use server";

import * as Sentry from "@sentry/nextjs";
import {
  buildRequestUrl,
  makeAuthenticatedFileUpload,
  makeAuthenticatedRequest,
} from "./helpers";
import { getAgptServerUrl } from "@/lib/env-config";

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
  baseUrl = getAgptServerUrl(),
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
  baseUrl = getAgptServerUrl(),
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
