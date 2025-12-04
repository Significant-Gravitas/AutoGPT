"use client";

import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";
import { Button } from "@/components/atoms/Button/Button";
import { CircleNotchIcon, FolderOpenIcon } from "@phosphor-icons/react";
import {
  Props as BaseProps,
  useGoogleDrivePicker,
} from "./useGoogleDrivePicker";

export type Props = BaseProps;

export function GoogleDrivePicker(props: Props) {
  const {
    credentials,
    hasGoogleOAuth,
    isAuthInProgress,
    isLoading,
    handleOpenPicker,
    selectedCredential,
    setSelectedCredential,
  } = useGoogleDrivePicker(props);

  if (!credentials || credentials.isLoading) {
    return <CircleNotchIcon className="size-6 animate-spin" />;
  }

  if (!hasGoogleOAuth) {
    return (
      <CredentialsInput
        schema={credentials.schema}
        selectedCredentials={selectedCredential}
        onSelectCredentials={setSelectedCredential}
        hideIfSingleCredentialAvailable
      />
    );
  }

  const hasMultipleCredentials =
    credentials.savedCredentials && credentials.savedCredentials.length > 1;

  return (
    <div className="flex flex-col gap-2">
      {hasMultipleCredentials && (
        <CredentialsInput
          schema={credentials.schema}
          selectedCredentials={selectedCredential}
          onSelectCredentials={setSelectedCredential}
          hideIfSingleCredentialAvailable={false}
        />
      )}
      <Button
        size="small"
        onClick={handleOpenPicker}
        disabled={props.disabled || isLoading || isAuthInProgress}
      >
        <FolderOpenIcon className="size-4" />
        {props.buttonText || "Choose file(s) from Google Drive"}
      </Button>
    </div>
  );
}
