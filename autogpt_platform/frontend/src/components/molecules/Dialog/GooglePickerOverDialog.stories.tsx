import { Button } from "@/components/atoms/Button/Button";
import { GoogleDrivePicker } from "@/components/contextual/GoogleDrivePicker/GoogleDrivePicker";
import {
  getGetV1GetSpecificCredentialByIdMockHandler,
  getPostV1GetPickerTokenMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { CredentialsMetaResponse as SavedCredential } from "@/lib/autogpt-server-api/types";
import {
  CredentialsProvidersContext,
  type CredentialsProviderData,
} from "@/providers/agent-credentials/credentials-provider";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ReactNode } from "react";
import { Dialog } from "./Dialog";
import {
  Dialog as PrimitiveDialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./components/DialogPrimitives";

// The Google Drive picker opened from inside each dialog implementation
// (molecules/Dialog and DialogPrimitives). The picker is the app's own
// GoogleDrivePicker: it loads Google's api.js and Identity Services and
// builds the picker with a token from the picker-token endpoint, which MSW
// answers here with a stub. globals.css lowers the open dialog while the
// picker is up, so the picker paints above it and takes the clicks.
const DRIVE_FILE = "https://www.googleapis.com/auth/drive.file";

const credential: CredentialsMetaResponse = {
  id: "story-google-credential",
  provider: "google",
  type: "oauth2",
  title: "story@example.com",
  scopes: [DRIVE_FILE],
  username: "story@example.com",
};

const savedCredential: SavedCredential = {
  id: credential.id,
  provider: "google",
  type: "oauth2",
  title: "story@example.com",
  scopes: [DRIVE_FILE],
  username: "story@example.com",
};

function rejectInStory(): never {
  throw new Error("Not available in this story");
}

const googleProvider: CredentialsProviderData = {
  provider: "google",
  providerName: "Google",
  savedCredentials: [savedCredential],
  isSystemProvider: false,
  oAuthCallback: rejectInStory,
  mcpOAuthCallback: rejectInStory,
  mcpStoreToken: rejectInStory,
  createAPIKeyCredentials: rejectInStory,
  createUserPasswordCredentials: rejectInStory,
  createHostScopedCredentials: rejectInStory,
  deleteCredentials: rejectInStory,
};

function Picker() {
  return (
    <GoogleDrivePicker
      scopes={[DRIVE_FILE]}
      developerKey="story-developer-key"
      clientId="story-client-id.apps.googleusercontent.com"
      appId="000000000000"
      requirePlatformCredentials
      onPicked={() => undefined}
      onCanceled={() => undefined}
      onError={() => undefined}
    />
  );
}

function WithGoogleCredentials({ children }: { children: ReactNode }) {
  return (
    <CredentialsProvidersContext.Provider value={{ google: googleProvider }}>
      {children}
    </CredentialsProvidersContext.Provider>
  );
}

const meta: Meta = {
  title: "Molecules/Dialog/Google picker over a dialog",
  parameters: {
    layout: "centered",
    msw: {
      handlers: [
        getGetV1GetSpecificCredentialByIdMockHandler(credential),
        getPostV1GetPickerTokenMockHandler({
          access_token: "story-access-token",
        }),
      ],
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const OverDialog: Story = {
  render: renderOverDialog,
};

export const OverDialogPrimitives: Story = {
  render: renderOverDialogPrimitives,
};

function renderOverDialog() {
  return (
    <WithGoogleCredentials>
      <Dialog title="Choose a file">
        <Dialog.Trigger>
          <Button variant="primary">Open dialog</Button>
        </Dialog.Trigger>
        <Dialog.Content>
          <Picker />
        </Dialog.Content>
      </Dialog>
    </WithGoogleCredentials>
  );
}

function renderOverDialogPrimitives() {
  return (
    <WithGoogleCredentials>
      <PrimitiveDialog>
        <DialogTrigger asChild>
          <Button variant="primary">Open dialog</Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Choose a file</DialogTitle>
          </DialogHeader>
          <Picker />
        </DialogContent>
      </PrimitiveDialog>
    </WithGoogleCredentials>
  );
}
