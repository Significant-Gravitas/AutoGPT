"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useGoogleDrivePicker } from "./useGoogleDrivePicker";

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

  async function handleClick() {
    await openPicker();
  }

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
