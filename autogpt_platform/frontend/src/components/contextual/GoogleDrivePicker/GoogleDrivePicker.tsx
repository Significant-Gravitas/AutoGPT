"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useGoogleDrivePicker } from "./useGoogleDrivePicker";
import useCredentials from "@/hooks/useCredentials";
import { useMemo } from "react";
import { CircleNotchIcon } from "@phosphor-icons/react";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";

const defaultScopes = ["https://www.googleapis.com/auth/drive.file"];

export type Props = {
  buttonText?: string;
  disabled?: boolean;
} & Parameters<typeof useGoogleDrivePicker>[0];

export function GoogleDrivePicker(props: Props) {
  const { isAuthInProgress, isLoading, handleOpenPicker } =
    useGoogleDrivePicker(props);

  const requestedScopes = props?.scopes || defaultScopes;

  const credentialsSchema: BlockIOCredentialsSubSchema = {
    type: "object",
    title: "Google Drive",
    description: "Google OAuth needed to access Google Drive",
    properties: {},
    required: [],
    credentials_provider: ["google"],
    credentials_types: ["oauth2"],
    credentials_scopes: requestedScopes,
    secret: true,
  };

  const credentials = useCredentials(credentialsSchema);

  const hasGoogleOAuth = useMemo(() => {
    if (!credentials || credentials.isLoading) return false;
    return credentials.savedCredentials?.length > 0;
  }, [credentials]);

  if (!credentials || credentials.isLoading) {
    return <CircleNotchIcon className="size-6 animate-spin" />;
  }

  if (!hasGoogleOAuth)
    return (
      <CredentialsInput
        schema={credentialsSchema}
        onSelectCredentials={() => {}}
        hideIfSingleCredentialAvailable
      />
    );

  return (
    <Button
      size="small"
      onClick={handleOpenPicker}
      disabled={props.disabled || isLoading || isAuthInProgress}
    >
      {props.buttonText || "Choose file from Google Drive"}
    </Button>
  );
}
