"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useGoogleDrivePicker } from "./useGoogleDrivePicker";
import useCredentials from "@/hooks/useCredentials";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import { useMemo } from "react";
import { OAuthLogin } from "./components/OAuthLogin/OAuthLogin";

export type GoogleDrivePickerProps = {
  buttonText?: string;
  disabled?: boolean;
} & Parameters<typeof useGoogleDrivePicker>[0];

export function GoogleDrivePicker(props: GoogleDrivePickerProps) {
  const {
    buttonText = "Choose from Google Drive",
    disabled,
    ...hookOptions
  } = props;
  const { openPicker, isAuthInProgress, isLoading } =
    useGoogleDrivePicker(hookOptions);

  const requestedScopes = useMemo(
    () => hookOptions?.scopes || ["https://www.googleapis.com/auth/drive.file"],
    [hookOptions?.scopes],
  );

  const credentialsSchema: BlockIOCredentialsSubSchema = useMemo(
    () => ({
      type: "object",
      title: "Google Drive",
      description: "Google OAuth needed to access Google Drive",
      properties: {},
      required: [],
      credentials_provider: ["google"],
      credentials_types: ["oauth2"],
      credentials_scopes: requestedScopes,
      secret: true,
    }),
    [requestedScopes],
  );

  const credentials = useCredentials(credentialsSchema);

  async function handleClick() {
    await openPicker();
  }

  const hasGoogleOAuth = useMemo(() => {
    if (!credentials || credentials.isLoading) return false;
    return credentials.savedCredentials?.length > 0;
  }, [credentials]);

  if (!credentials || credentials.isLoading)
    return (
      <Button size="small" disabled>
        {buttonText}
      </Button>
    );

  if (!hasGoogleOAuth)
    return (
      <OAuthLogin
        provider={credentials.provider}
        providerName={credentials.providerName}
        scopes={credentials.schema.credentials_scopes}
        disabled={disabled}
        oAuthCallback={credentials.oAuthCallback}
      />
    );

  return (
    <Button
      size="small"
      onClick={handleClick}
      disabled={disabled || isLoading || isAuthInProgress}
    >
      {buttonText}
    </Button>
  );
}
