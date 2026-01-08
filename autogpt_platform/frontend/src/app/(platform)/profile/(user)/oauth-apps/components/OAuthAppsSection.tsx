"use client";

import { useRef, useState } from "react";
import {
  UploadIcon,
  ImageIcon,
  PowerIcon,
  TrashIcon,
  KeyIcon,
  PlusIcon,
  CopyIcon,
  EyeIcon,
  EyeSlashIcon,
} from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Badge } from "@/components/atoms/Badge/Badge";
import { useOAuthApps } from "./useOAuthApps";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/__legacy__/ui/alert-dialog";
import { Input } from "@/components/atoms/Input/Input";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import type { OAuthApplicationCreationResult } from "@/lib/autogpt-server-api/types";

const AVAILABLE_SCOPES = [
  {
    value: "EXECUTE_GRAPH",
    label: "Execute Graph",
    description: "Run agent graphs",
  },
  { value: "READ_GRAPH", label: "Read Graph", description: "Read agent graphs" },
  {
    value: "EXECUTE_BLOCK",
    label: "Execute Block",
    description: "Execute individual blocks",
  },
  {
    value: "READ_BLOCK",
    label: "Read Block",
    description: "Read block definitions",
  },
  { value: "READ_STORE", label: "Read Store", description: "Access the store" },
  { value: "USE_TOOLS", label: "Use Tools", description: "Use available tools" },
  {
    value: "MANAGE_INTEGRATIONS",
    label: "Manage Integrations",
    description: "Manage integrations",
  },
  {
    value: "READ_INTEGRATIONS",
    label: "Read Integrations",
    description: "Read integrations",
  },
  {
    value: "DELETE_INTEGRATIONS",
    label: "Delete Integrations",
    description: "Delete integrations",
  },
];

export function OAuthAppsSection() {
  const {
    oauthApps,
    isLoading,
    updatingAppId,
    uploadingAppId,
    deletingAppId,
    regeneratingAppId,
    isCreating,
    handleToggleStatus,
    handleUploadLogo,
    handleCreateApp,
    handleDeleteApp,
    handleRegenerateSecret,
  } = useOAuthApps();

  const fileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({});

  // Create dialog state
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [createStep, setCreateStep] = useState<"form" | "success">("form");
  const [creationResult, setCreationResult] =
    useState<OAuthApplicationCreationResult | null>(null);
  const [isSecretVisible, setIsSecretVisible] = useState(false);

  // Form state
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [redirectUris, setRedirectUris] = useState<string[]>([""]);
  const [selectedScopes, setSelectedScopes] = useState<string[]>([]);
  const [formError, setFormError] = useState<string | null>(null);

  // Delete dialog state
  const [deleteAppId, setDeleteAppId] = useState<string | null>(null);
  const [deleteAppName, setDeleteAppName] = useState<string>("");

  // Regenerate secret dialog state
  const [regenerateAppId, setRegenerateAppId] = useState<string | null>(null);
  const [regenerateAppName, setRegenerateAppName] = useState<string>("");
  const [newSecret, setNewSecret] = useState<string | null>(null);

  const resetCreateForm = () => {
    setCreateStep("form");
    setCreationResult(null);
    setIsSecretVisible(false);
    setName("");
    setDescription("");
    setRedirectUris([""]);
    setSelectedScopes([]);
    setFormError(null);
  };

  const handleCloseCreateDialog = () => {
    resetCreateForm();
    setIsCreateDialogOpen(false);
  };

  const handleFileChange = (
    appId: string,
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      handleUploadLogo(appId, file);
    }
    event.target.value = "";
  };

  const addRedirectUri = () => {
    setRedirectUris([...redirectUris, ""]);
  };

  const removeRedirectUri = (index: number) => {
    if (redirectUris.length > 1) {
      setRedirectUris(redirectUris.filter((_, i) => i !== index));
    }
  };

  const updateRedirectUri = (index: number, value: string) => {
    const updated = [...redirectUris];
    updated[index] = value;
    setRedirectUris(updated);
  };

  const toggleScope = (scope: string) => {
    setSelectedScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope],
    );
  };

  const handleSubmitCreate = async () => {
    setFormError(null);

    const validUris = redirectUris.filter((uri) => uri.trim());
    if (!name.trim()) {
      setFormError("Name is required");
      return;
    }
    if (validUris.length === 0) {
      setFormError("At least one redirect URI is required");
      return;
    }
    if (selectedScopes.length === 0) {
      setFormError("At least one scope is required");
      return;
    }

    const result = await handleCreateApp({
      name: name.trim(),
      description: description.trim() || undefined,
      redirect_uris: validUris,
      scopes: selectedScopes,
    });

    if (result) {
      setCreationResult(result);
      setCreateStep("success");
    }
  };

  const handleConfirmDelete = async () => {
    if (deleteAppId) {
      await handleDeleteApp(deleteAppId);
      setDeleteAppId(null);
      setDeleteAppName("");
    }
  };

  const handleConfirmRegenerate = async () => {
    if (regenerateAppId) {
      const secret = await handleRegenerateSecret(regenerateAppId);
      if (secret) {
        setNewSecret(secret);
      }
    }
  };

  const handleCloseRegenerateDialog = () => {
    setRegenerateAppId(null);
    setRegenerateAppName("");
    setNewSecret(null);
    setIsSecretVisible(false);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (isLoading) {
    return (
      <div className="flex justify-center p-4">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Create Button */}
      <div className="flex justify-end">
        <Button
          variant="primary"
          onClick={() => setIsCreateDialogOpen(true)}
          leftIcon={<PlusIcon className="h-4 w-4" />}
        >
          Create OAuth App
        </Button>
      </div>

      {/* Empty State */}
      {oauthApps.length === 0 ? (
        <div className="py-8 text-center text-muted-foreground">
          <p>You don&apos;t have any OAuth applications yet.</p>
          <p className="mt-2 text-sm">
            Click &quot;Create OAuth App&quot; to register your first
            application.
          </p>
        </div>
      ) : (
        /* App Cards Grid */
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
                    <ImageIcon className="h-7 w-7 text-muted-foreground" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="truncate text-lg font-semibold">
                      {app.name}
                    </h3>
                    <Badge
                      className="ml-2"
                      variant={app.is_active ? "success" : "error"}
                    >
                      {app.is_active ? "Active" : "Disabled"}
                    </Badge>
                  </div>
                  {app.description && (
                    <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                      {app.description}
                    </p>
                  )}
                </div>
              </div>

              {/* Client ID */}
              <div>
                <span className="text-xs font-medium text-muted-foreground">
                  Client ID
                </span>
                <div className="mt-1 flex items-center gap-2">
                  <code
                    data-testid="oauth-app-client-id"
                    className="block flex-1 truncate rounded-md border bg-muted px-3 py-2 text-xs"
                  >
                    {app.client_id}
                  </code>
                  <Button
                    variant="ghost"
                    size="small"
                    onClick={() => copyToClipboard(app.client_id)}
                  >
                    <CopyIcon className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Footer: Created date and Actions */}
              <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
                <span className="text-xs text-muted-foreground">
                  Created {new Date(app.created_at).toLocaleDateString()}
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    variant={app.is_active ? "outline" : "primary"}
                    size="small"
                    onClick={() => handleToggleStatus(app.id, app.is_active)}
                    loading={updatingAppId === app.id}
                    leftIcon={<PowerIcon className="h-4 w-4" />}
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
                    size="small"
                    onClick={() => fileInputRefs.current[app.id]?.click()}
                    loading={uploadingAppId === app.id}
                    leftIcon={<UploadIcon className="h-4 w-4" />}
                  >
                    Logo
                  </Button>
                  <Button
                    variant="outline"
                    size="small"
                    onClick={() => {
                      setRegenerateAppId(app.id);
                      setRegenerateAppName(app.name);
                    }}
                    loading={regeneratingAppId === app.id}
                    leftIcon={<KeyIcon className="h-4 w-4" />}
                  >
                    Secret
                  </Button>
                  <Button
                    variant="outline"
                    size="small"
                    onClick={() => {
                      setDeleteAppId(app.id);
                      setDeleteAppName(app.name);
                    }}
                    loading={deletingAppId === app.id}
                    leftIcon={<TrashIcon className="h-4 w-4" />}
                    className="text-red-600 hover:bg-red-50"
                  >
                    Delete
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create OAuth App Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={handleCloseCreateDialog}>
        <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
          {createStep === "form" ? (
            <>
              <DialogHeader>
                <DialogTitle>Create OAuth Application</DialogTitle>
                <DialogDescription>
                  Create a new OAuth application for third-party integrations.
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-6">
                {formError && (
                  <div className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                    {formError}
                  </div>
                )}

                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Application Name *
                    </label>
                    <Input
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="My Application"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Description</label>
                    <Textarea
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Describe what this application does..."
                      rows={2}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">
                        Redirect URIs *
                      </label>
                      <Button
                        type="button"
                        variant="ghost"
                        size="small"
                        onClick={addRedirectUri}
                      >
                        <PlusIcon className="mr-1 h-3 w-3" />
                        Add URI
                      </Button>
                    </div>
                    <div className="space-y-2">
                      {redirectUris.map((uri, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <Input
                            value={uri}
                            onChange={(e) =>
                              updateRedirectUri(index, e.target.value)
                            }
                            placeholder="https://example.com/callback"
                          />
                          {redirectUris.length > 1 && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="small"
                              onClick={() => removeRedirectUri(index)}
                            >
                              <TrashIcon className="h-4 w-4" />
                            </Button>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Scopes *</label>
                    <div className="grid grid-cols-2 gap-2 rounded-md border bg-muted/30 p-4">
                      {AVAILABLE_SCOPES.map((scope) => (
                        <div
                          key={scope.value}
                          className="flex items-start space-x-2"
                        >
                          <Checkbox
                            id={scope.value}
                            checked={selectedScopes.includes(scope.value)}
                            onCheckedChange={() => toggleScope(scope.value)}
                          />
                          <div className="leading-none">
                            <label
                              htmlFor={scope.value}
                              className="cursor-pointer text-sm font-medium"
                            >
                              {scope.label}
                            </label>
                            <p className="text-xs text-muted-foreground">
                              {scope.description}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <DialogFooter>
                <Button variant="secondary" onClick={handleCloseCreateDialog}>
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  onClick={handleSubmitCreate}
                  loading={isCreating}
                >
                  Create Application
                </Button>
              </DialogFooter>
            </>
          ) : (
            <>
              <DialogHeader>
                <DialogTitle>OAuth Application Created</DialogTitle>
                <DialogDescription>
                  Your OAuth application has been created successfully. Save the
                  client secret now - it will only be shown once!
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4">
                <div className="rounded-md border border-green-200 bg-green-50 p-4">
                  <h4 className="mb-2 font-medium text-green-800">
                    {creationResult?.application.name}
                  </h4>
                  <p className="text-sm text-green-700">
                    Application created successfully
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="space-y-1">
                    <label className="text-sm text-muted-foreground">
                      Client ID
                    </label>
                    <div className="flex items-center gap-2">
                      <code className="block flex-1 break-all rounded bg-muted px-3 py-2 text-sm">
                        {creationResult?.application.client_id}
                      </code>
                      <Button
                        variant="ghost"
                        size="small"
                        onClick={() =>
                          copyToClipboard(
                            creationResult?.application.client_id || "",
                          )
                        }
                      >
                        <CopyIcon className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2 rounded-md border border-yellow-200 bg-yellow-50 p-4">
                    <label className="text-sm font-medium text-yellow-800">
                      Client Secret (Save this now!)
                    </label>
                    <div className="flex items-center gap-2">
                      <code className="block flex-1 break-all rounded border bg-white px-3 py-2 text-sm">
                        {isSecretVisible
                          ? creationResult?.client_secret_plaintext
                          : "••••••••••••••••••••••••••••••••"}
                      </code>
                      <Button
                        variant="ghost"
                        size="small"
                        onClick={() => setIsSecretVisible(!isSecretVisible)}
                      >
                        {isSecretVisible ? (
                          <EyeSlashIcon className="h-4 w-4" />
                        ) : (
                          <EyeIcon className="h-4 w-4" />
                        )}
                      </Button>
                      <Button
                        variant="ghost"
                        size="small"
                        onClick={() =>
                          copyToClipboard(
                            creationResult?.client_secret_plaintext || "",
                          )
                        }
                      >
                        <CopyIcon className="h-4 w-4" />
                      </Button>
                    </div>
                    <p className="text-xs text-yellow-700">
                      This secret will only be shown once. Store it securely!
                    </p>
                  </div>
                </div>
              </div>

              <DialogFooter>
                <Button variant="primary" onClick={handleCloseCreateDialog}>
                  Done
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog
        open={deleteAppId !== null}
        onOpenChange={(open) => {
          if (!open) {
            setDeleteAppId(null);
            setDeleteAppName("");
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete OAuth Application</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete <strong>{deleteAppName}</strong>?
              This will also delete all associated authorization codes, access
              tokens, and refresh tokens. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleConfirmDelete}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Regenerate Secret Dialog */}
      <Dialog
        open={regenerateAppId !== null}
        onOpenChange={(open) => {
          if (!open) {
            handleCloseRegenerateDialog();
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {newSecret ? "New Client Secret Generated" : "Regenerate Secret"}
            </DialogTitle>
            <DialogDescription>
              {newSecret
                ? "Your new client secret is shown below. Make sure to copy it now - you won't be able to see it again!"
                : `Are you sure you want to regenerate the client secret for "${regenerateAppName}"? The old secret will be invalidated immediately.`}
            </DialogDescription>
          </DialogHeader>

          {newSecret && (
            <div className="space-y-4">
              <div className="space-y-2 rounded-md border border-yellow-200 bg-yellow-50 p-4">
                <label className="text-sm font-medium text-yellow-800">
                  Client Secret
                </label>
                <div className="flex items-center gap-2">
                  <code className="block flex-1 break-all rounded border bg-white px-3 py-2 text-sm">
                    {isSecretVisible
                      ? newSecret
                      : "••••••••••••••••••••••••••••••••"}
                  </code>
                  <Button
                    variant="ghost"
                    size="small"
                    onClick={() => setIsSecretVisible(!isSecretVisible)}
                  >
                    {isSecretVisible ? (
                      <EyeSlashIcon className="h-4 w-4" />
                    ) : (
                      <EyeIcon className="h-4 w-4" />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="small"
                    onClick={() => copyToClipboard(newSecret)}
                  >
                    <CopyIcon className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-xs text-yellow-700">
                  This secret will only be shown once. Store it securely!
                </p>
              </div>
            </div>
          )}

          <DialogFooter>
            {newSecret ? (
              <Button
                variant="primary"
                onClick={handleCloseRegenerateDialog}
              >
                Done
              </Button>
            ) : (
              <>
                <Button
                  variant="secondary"
                  onClick={handleCloseRegenerateDialog}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  onClick={handleConfirmRegenerate}
                  loading={regeneratingAppId !== null && !newSecret}
                >
                  Regenerate Secret
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
