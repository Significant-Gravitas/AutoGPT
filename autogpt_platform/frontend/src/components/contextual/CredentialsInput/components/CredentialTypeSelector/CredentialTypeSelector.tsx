"use client";

import {
  Form,
  FormDescription,
  FormField,
} from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
  CredentialsType,
} from "@/lib/autogpt-server-api/types";
import { useAPIKeyCredentialsModal } from "../APIKeyCredentialsModal/useAPIKeyCredentialsModal";
import { getCredentialTypeIcon, getCredentialTypeLabel } from "../../helpers";

type Props = {
  schema: BlockIOCredentialsSubSchema;
  open: boolean;
  onClose: () => void;
  provider: string;
  providerName: string;
  supportedTypes: CredentialsType[];
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
  onOAuthLogin: () => void;
  onOpenPasswordModal: () => void;
  onOpenHostScopedModal: () => void;
  siblingInputs?: Record<string, unknown>;
};

export function CredentialTypeSelector({
  schema,
  open,
  onClose,
  provider,
  providerName,
  supportedTypes,
  onCredentialsCreate,
  onOAuthLogin,
  onOpenPasswordModal,
  onOpenHostScopedModal,
  siblingInputs,
}: Props) {
  const defaultTab = supportedTypes[0];

  return (
    <Dialog
      title={`Add credential for ${providerName}`}
      controlled={{
        isOpen: open,
        set: (isOpen) => {
          if (!isOpen) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "28rem" }}
    >
      <Dialog.Content>
        <TabsLine defaultValue={defaultTab}>
          <TabsLineList>
            {supportedTypes.map((type) => {
              const Icon = getCredentialTypeIcon(type, provider);
              return (
                <TabsLineTrigger
                  key={type}
                  value={type}
                  className="inline-flex items-center gap-1.5"
                >
                  <Icon size={16} />
                  {getCredentialTypeLabel(type)}
                </TabsLineTrigger>
              );
            })}
          </TabsLineList>

          {supportedTypes.includes("oauth2") && (
            <TabsLineContent value="oauth2">
              <OAuthTabContent
                providerName={providerName}
                onOAuthLogin={() => {
                  onClose();
                  onOAuthLogin();
                }}
              />
            </TabsLineContent>
          )}

          {supportedTypes.includes("api_key") && (
            <TabsLineContent value="api_key">
              <APIKeyTabContent
                schema={schema}
                siblingInputs={siblingInputs}
                onCredentialsCreate={(creds) => {
                  onCredentialsCreate(creds);
                  onClose();
                }}
              />
            </TabsLineContent>
          )}

          {supportedTypes.includes("user_password") && (
            <TabsLineContent value="user_password">
              <SimpleActionTab
                description="Add a username and password credential."
                buttonLabel="Enter credentials"
                onClick={() => {
                  onClose();
                  onOpenPasswordModal();
                }}
              />
            </TabsLineContent>
          )}

          {supportedTypes.includes("host_scoped") && (
            <TabsLineContent value="host_scoped">
              <SimpleActionTab
                description="Add host-scoped headers for authentication."
                buttonLabel="Add headers"
                onClick={() => {
                  onClose();
                  onOpenHostScopedModal();
                }}
              />
            </TabsLineContent>
          )}
        </TabsLine>
      </Dialog.Content>
    </Dialog>
  );
}

type OAuthTabContentProps = {
  providerName: string;
  onOAuthLogin: () => void;
};

function OAuthTabContent({ providerName, onOAuthLogin }: OAuthTabContentProps) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-zinc-600">
        Sign in with your {providerName} account using OAuth.
      </p>
      <Button
        variant="primary"
        size="small"
        onClick={onOAuthLogin}
        type="button"
      >
        Sign in with {providerName}
      </Button>
    </div>
  );
}

type APIKeyTabContentProps = {
  schema: BlockIOCredentialsSubSchema;
  siblingInputs?: Record<string, unknown>;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
};

function APIKeyTabContent({
  schema,
  siblingInputs,
  onCredentialsCreate,
}: APIKeyTabContentProps) {
  const {
    form,
    isLoading,
    isSubmitting,
    supportsApiKey,
    schemaDescription,
    onSubmit,
  } = useAPIKeyCredentialsModal({ schema, siblingInputs, onCredentialsCreate });

  if (!supportsApiKey && !isLoading) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-4 w-3/4" />
        <div className="space-y-2">
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-68" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {schemaDescription && (
        <p className="text-sm text-zinc-600">{schemaDescription}</p>
      )}

      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-2">
          <FormField
            control={form.control}
            name="title"
            render={({ field }) => (
              <Input
                id="title"
                label="Name"
                type="text"
                placeholder="Enter a name for this API Key..."
                {...field}
              />
            )}
          />
          <FormField
            control={form.control}
            name="apiKey"
            render={({ field }) => (
              <Input
                id="apiKey"
                label="API Key"
                type="password"
                placeholder="Enter API Key..."
                hint={
                  schema.credentials_scopes ? (
                    <FormDescription>
                      Required scope(s) for this block:{" "}
                      {schema.credentials_scopes?.map((s, i, a) => (
                        <span key={i}>
                          <code className="text-xs font-bold">{s}</code>
                          {i < a.length - 1 && ", "}
                        </span>
                      ))}
                    </FormDescription>
                  ) : null
                }
                {...field}
              />
            )}
          />
          <FormField
            control={form.control}
            name="expiresAt"
            render={({ field }) => (
              <Input
                id="expiresAt"
                label="Expiration Date"
                type="datetime-local"
                placeholder="Select expiration date..."
                value={field.value}
                onChange={(e) => field.onChange(e.target.value)}
                onBlur={field.onBlur}
                name={field.name}
              />
            )}
          />
          <Button
            type="submit"
            className="min-w-68"
            loading={isSubmitting}
            disabled={isSubmitting}
          >
            Add API Key
          </Button>
        </form>
      </Form>
    </div>
  );
}

type SimpleActionTabProps = {
  description: string;
  buttonLabel: string;
  onClick: () => void;
};

function SimpleActionTab({
  description,
  buttonLabel,
  onClick,
}: SimpleActionTabProps) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-zinc-600">{description}</p>
      <Button variant="primary" size="small" onClick={onClick} type="button">
        {buttonLabel}
      </Button>
    </div>
  );
}
