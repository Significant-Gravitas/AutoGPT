"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AuthDivider } from "@/components/auth/AuthSplitLayout/AuthDivider";
import { FlaskIcon } from "@phosphor-icons/react/dist/ssr";
import { PREVIEW_ROLES } from "./helpers";
import { usePreviewLoginButtons } from "./usePreviewLoginButtons";

export function PreviewLoginButtons() {
  const {
    isPreview,
    isConfigured,
    isCheckingConfig,
    loadingRole,
    handlePreviewLogin,
  } = usePreviewLoginButtons();

  if (!isPreview) {
    return null;
  }

  return (
    <div className="mt-2 flex w-full flex-col gap-3">
      <AuthDivider />

      <div className="flex items-center gap-2">
        <FlaskIcon size={16} weight="duotone" className="text-slate-500" />
        <Text variant="small-medium" className="!text-slate-500">
          Preview test accounts
        </Text>
      </div>

      {!isCheckingConfig && !isConfigured ? (
        <Text variant="small" className="!text-slate-500">
          Set PREVIEW_ACCOUNTS_PASSWORD in the preview environment to enable
          one-click sign-in.
        </Text>
      ) : null}

      <div className="grid grid-cols-2 gap-2">
        {PREVIEW_ROLES.map(({ role, label }) => (
          <Button
            key={role}
            variant="secondary"
            size="small"
            className="w-full"
            disabled={!isConfigured || loadingRole !== null}
            loading={loadingRole === role}
            onClick={() => handlePreviewLogin(role)}
          >
            {label}
          </Button>
        ))}
      </div>
    </div>
  );
}
