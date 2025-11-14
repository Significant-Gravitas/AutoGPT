"use client";

import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import { Button } from "@/components/atoms/Button/Button";
import { CircleNotchIcon, FolderOpenIcon } from "@phosphor-icons/react";
import { Props, useGoogleDrivePicker } from "./useGoogleDrivePicker";

export function GoogleDrivePicker(props: Props) {
  const {
    credentials,
    hasGoogleOAuth,
    isAuthInProgress,
    isLoading,
    handleOpenPicker,
  } = useGoogleDrivePicker(props);

  if (!credentials || credentials.isLoading) {
    return <CircleNotchIcon className="size-6 animate-spin" />;
  }

  if (!hasGoogleOAuth)
    return (
      <CredentialsInput
        schema={credentials.schema}
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
      <FolderOpenIcon className="size-4" />
      {props.buttonText || "Choose file(s) from Google Drive"}
    </Button>
  );
}
