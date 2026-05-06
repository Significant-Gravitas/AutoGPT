import {
  useGetPlatformLinkingGetDisplayInfoForALinkToken,
  usePostPlatformLinkingConfirmAServerLinkTokenUserMustBeAuthenticated,
  usePostPlatformLinkingConfirmAUserLinkTokenUserMustBeAuthenticated,
} from "@/app/api/__generated__/endpoints/platform-linking/platform-linking";
import { ConfirmLinkResponse } from "@/app/api/__generated__/models/confirmLinkResponse";
import { ConfirmUserLinkResponse } from "@/app/api/__generated__/models/confirmUserLinkResponse";
import { LinkType } from "@/app/api/__generated__/models/linkType";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useParams, useSearchParams } from "next/navigation";
import {
  getLoginRedirect,
  getPlatformDisplayName,
  isUserLink,
  TOKEN_PATTERN,
} from "./helpers";

export type PageStatus =
  | "loading"
  | "not-authenticated"
  | "ready"
  | "linking"
  | "success"
  | "error";

interface ViewData {
  linkType: LinkType;
  platform: string;
  serverName: string | null;
}

export function usePlatformLinkingPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const rawToken = (params.token as string | undefined) ?? "";
  const token = TOKEN_PATTERN.test(rawToken) ? rawToken : null;
  const platformFromUrl = getPlatformDisplayName(searchParams.get("platform"));
  const { user, isUserLoading, logOut } = useSupabase();

  const {
    data: info,
    isLoading: isInfoLoading,
    isError: isInfoError,
  } = useGetPlatformLinkingGetDisplayInfoForALinkToken(token ?? "", {
    query: {
      enabled: Boolean(token) && Boolean(user),
      select: (res) => (res && res.status === 200 ? res.data : undefined),
      retry: false,
    },
  });

  const serverConfirm =
    usePostPlatformLinkingConfirmAServerLinkTokenUserMustBeAuthenticated();
  const userConfirm =
    usePostPlatformLinkingConfirmAUserLinkTokenUserMustBeAuthenticated();

  const mutation =
    info && isUserLink(info.link_type) ? userConfirm : serverConfirm;
  const confirmResponse =
    mutation.data && mutation.data.status >= 200 && mutation.data.status < 300
      ? (mutation.data.data as ConfirmLinkResponse | ConfirmUserLinkResponse)
      : undefined;

  const status = resolveStatus({
    hasToken: Boolean(token),
    isUserLoading,
    hasUser: Boolean(user),
    isInfoLoading,
    isInfoError,
    isMutating: mutation.isPending,
    isMutationSuccess: mutation.isSuccess,
    isMutationError: mutation.isError,
  });

  function handleLink() {
    if (!token || !info) return;
    mutation.mutate({ token });
  }

  async function handleSwitchAccount() {
    await logOut();
    window.location.href = getLoginRedirect(token);
  }

  return {
    status,
    token,
    loginRedirect: getLoginRedirect(token),
    userEmail: user?.email ?? null,
    viewData: buildViewData({ info, platformFromUrl }),
    successData: buildSuccessData({
      confirmResponse,
      fallbackPlatform: platformFromUrl,
    }),
    errorMessage: buildErrorMessage({
      hasToken: Boolean(token),
      isInfoError,
      mutation,
    }),
    handleLink,
    handleSwitchAccount,
  };
}

function resolveStatus(args: {
  hasToken: boolean;
  isUserLoading: boolean;
  hasUser: boolean;
  isInfoLoading: boolean;
  isInfoError: boolean;
  isMutating: boolean;
  isMutationSuccess: boolean;
  isMutationError: boolean;
}): PageStatus {
  if (!args.hasToken) return "error";
  if (args.isUserLoading) return "loading";
  if (!args.hasUser) return "not-authenticated";
  if (args.isMutationSuccess) return "success";
  if (args.isMutationError) return "error";
  if (args.isMutating) return "linking";
  if (args.isInfoLoading) return "loading";
  if (args.isInfoError) return "error";
  return "ready";
}

function buildViewData(args: {
  info:
    | { link_type: LinkType; platform: string; server_name?: string | null }
    | undefined;
  platformFromUrl: string;
}): ViewData | null {
  if (!args.info) return null;
  return {
    linkType: args.info.link_type,
    platform:
      getPlatformDisplayName(args.info.platform) || args.platformFromUrl,
    serverName: args.info.server_name ?? null,
  };
}

function buildSuccessData(args: {
  confirmResponse: ConfirmLinkResponse | ConfirmUserLinkResponse | undefined;
  fallbackPlatform: string;
}): (ViewData & { platform: string }) | null {
  if (!args.confirmResponse) return null;
  const serverName =
    "server_name" in args.confirmResponse
      ? (args.confirmResponse.server_name ?? null)
      : null;
  return {
    linkType: args.confirmResponse.link_type ?? LinkType.SERVER,
    platform:
      getPlatformDisplayName(args.confirmResponse.platform) ||
      args.fallbackPlatform,
    serverName,
  };
}

function buildErrorMessage(args: {
  hasToken: boolean;
  isInfoError: boolean;
  mutation: { error: unknown };
}): string {
  if (!args.hasToken) {
    return "This setup link is malformed. Ask the bot for a new one.";
  }
  if (args.mutation.error) {
    return extractDetail(args.mutation.error) ?? DEFAULT_MUTATION_ERROR;
  }
  if (args.isInfoError) {
    return "Couldn't load setup details. The link may have expired.";
  }
  return DEFAULT_MUTATION_ERROR;
}

const DEFAULT_MUTATION_ERROR =
  "Failed to complete setup. The link may have expired.";

function extractDetail(error: unknown): string | undefined {
  if (!error || typeof error !== "object") return undefined;
  const responseDetail = extractDetailFromPayload(
    (error as { response?: unknown }).response,
  );
  if (responseDetail) return responseDetail;

  const dataDetail = extractDetailFromPayload(
    (error as { data?: unknown }).data,
  );
  if (dataDetail) return dataDetail;

  const message = (error as { message?: unknown }).message;
  return typeof message === "string" ? message : undefined;
}

function extractDetailFromPayload(payload: unknown): string | undefined {
  if (!payload || typeof payload !== "object") return undefined;
  const detail = (payload as { detail?: unknown }).detail;
  return typeof detail === "string" ? detail : undefined;
}
