"use server";

import { environment } from "@/services/environment";
import { login } from "../../actions";
import { PreviewRole } from "./helpers";

const PREVIEW_ACCOUNT_EMAILS: Record<PreviewRole, string> = {
  admin: "preview-admin@agpt.co",
  existing: "preview-existing@agpt.co",
  clean: "preview-clean@agpt.co",
  pro: "preview-pro@agpt.co",
  enterprise: "preview-enterprise@agpt.co",
};

function isPreviewEnvironment() {
  return Boolean(environment.getPreviewStealingDev());
}

export async function isPreviewLoginConfigured() {
  return (
    isPreviewEnvironment() && Boolean(process.env.PREVIEW_ACCOUNTS_PASSWORD)
  );
}

export async function loginAsPreviewAccount(role: PreviewRole) {
  // Gate the action server-side on the preview marker: it must be a no-op in any
  // non-preview environment, regardless of what the client sends. Hiding the UI
  // client-side is not enough since a server action is directly callable.
  if (!isPreviewEnvironment()) {
    return { success: false, error: "Preview login is not available" };
  }

  const email = PREVIEW_ACCOUNT_EMAILS[role];
  if (!email) {
    return { success: false, error: "Unknown preview account" };
  }

  const password = process.env.PREVIEW_ACCOUNTS_PASSWORD;
  if (!password) {
    return { success: false, error: "Preview login is not configured" };
  }

  return login(email, password);
}
