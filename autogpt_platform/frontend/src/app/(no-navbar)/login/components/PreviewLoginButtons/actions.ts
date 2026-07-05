"use server";

import { login } from "../../actions";

const PREVIEW_ACCOUNT_EMAILS: Record<string, string> = {
  admin: "preview-admin@agpt.co",
  existing: "preview-existing@agpt.co",
  clean: "preview-clean@agpt.co",
  pro: "preview-pro@agpt.co",
  enterprise: "preview-enterprise@agpt.co",
};

export async function isPreviewLoginConfigured() {
  return Boolean(process.env.PREVIEW_ACCOUNTS_PASSWORD);
}

export async function loginAsPreviewAccount(role: string) {
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
