"use client";

import { useRef } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { useOAuthApps } from "./useOAuthApps";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  Image01Icon,
  PowerIcon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function OAuthAppsSection() {
  const {
    oauthApps,
    isLoading,
    updatingAppId,
    uploadingAppId,
    handleToggleStatus,
    handleUploadLogo,
  } = useOAuthApps();

  const fileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({});

  const handleFileChange = (
    appId: string,
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      handleUploadLogo(appId, file);
    }
    // Reset the input so the same file can be selected again
    event.target.value = "";
  };

  if (isLoading) {
    return (
      <div className="flex justify-center p-4">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (oauthApps.length === 0) {
    return (
      <div className="py-8 text-center text-muted-foreground">
        <p>You don&apos;t have any OAuth applications.</p>
        <Text variant="body" tone="muted" className="mt-2">
          OAuth applications can currently <strong>not</strong> be registered
          via the API. Contact the system administrator to request an OAuth app
          registration.
        </Text>
      </div>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-1 lg:grid-cols-2">
      {oauthApps.map((app) => (
        <div
          key={app.id}
          data-testid="oauth-app-card"
          className="flex flex-col gap-4 rounded-xl border bg-card p-5"
        >
          {/* Header: Logo, Name, Status */}
          <div className="flex items-start gap-4">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center overflow-hidden rounded-xl border bg-muted">
              {app.logo_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={app.logo_url}
                  alt={`${app.name} logo`}
                  className="h-full w-full object-cover"
                />
              ) : (
                <Icon
                  icon={Image01Icon}
                  className="h-7 w-7 text-muted-foreground"
                />
              )}
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <Text variant="h5" as="h3" className="truncate" unmask={false}>
                  {app.name}
                </Text>
                <Badge
                  className="ml-2"
                  variant={app.is_active ? "success" : "error"}
                >
                  {app.is_active ? "Active" : "Disabled"}
                </Badge>
              </div>
              {app.description && (
                <Text
                  variant="body"
                  tone="muted"
                  className="mt-1 line-clamp-2"
                  unmask={false}
                >
                  {app.description}
                </Text>
              )}
            </div>
          </div>

          {/* Client ID */}
          <div>
            <span className="text-xs font-medium text-muted-foreground">
              Client ID
            </span>
            <code
              data-testid="oauth-app-client-id"
              className="mt-1 block w-full truncate rounded-md border bg-muted px-3 py-2 text-xs"
            >
              {app.client_id}
            </code>
          </div>

          {/* Footer: Created date and Actions */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
            <span className="text-xs text-muted-foreground">
              Created {new Date(app.created_at).toLocaleDateString()}
            </span>
            <div className="flex items-center gap-3">
              <Button
                variant={app.is_active ? "outline" : "primary"}
                size="md"
                onClick={() => handleToggleStatus(app.id, app.is_active)}
                loading={updatingAppId === app.id}
                leadingIcon={PowerIcon}
              >
                {app.is_active ? "Disable" : "Enable"}
              </Button>
              <input
                type="file"
                ref={(el) => {
                  fileInputRefs.current[app.id] = el;
                }}
                onChange={(e) => handleFileChange(app.id, e)}
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
              />
              <Button
                variant="outline"
                size="md"
                onClick={() => fileInputRefs.current[app.id]?.click()}
                loading={uploadingAppId === app.id}
                leadingIcon={Upload01Icon}
              >
                {app.logo_url ? "Change " : "Upload "}Logo
              </Button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
