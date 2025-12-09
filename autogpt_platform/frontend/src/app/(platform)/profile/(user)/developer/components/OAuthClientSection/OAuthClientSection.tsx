"use client";

import { Loader2, MoreVertical } from "lucide-react";
import { LuCopy } from "react-icons/lu";
import { Button } from "@/components/__legacy__/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import { Badge } from "@/components/__legacy__/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/__legacy__/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { Label } from "@/components/__legacy__/ui/label";
import { Input } from "@/components/__legacy__/ui/input";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { useOAuthClientSection } from "./useOAuthClientSection";

export function OAuthClientSection() {
  const {
    oauthClients,
    isLoading,
    isDeleting,
    isSuspending,
    isActivating,
    isRotatingWebhookSecret,
    isUpdating,
    handleDeleteClient,
    handleSuspendClient,
    handleActivateClient,
    handleRotateWebhookSecret,
    handleCopyWebhookSecret,
    handleEditClient,
    handleSaveClient,
    webhookSecretDialogOpen,
    setWebhookSecretDialogOpen,
    newWebhookSecret,
    editDialogOpen,
    setEditDialogOpen,
    editingClient,
    editFormState,
    setEditFormState,
  } = useOAuthClientSection();

  const isActionPending =
    isDeleting ||
    isSuspending ||
    isActivating ||
    isRotatingWebhookSecret ||
    isUpdating;

  return (
    <>
      {isLoading ? (
        <div className="flex justify-center p-4">
          <Loader2 className="h-6 w-6 animate-spin" />
        </div>
      ) : oauthClients && oauthClients.length > 0 ? (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Client ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Created</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {oauthClients.map((client) => (
              <TableRow key={client.id} data-testid="oauth-client-row">
                <TableCell>
                  <div className="flex flex-col">
                    <span className="font-medium">{client.name}</span>
                    {client.description && (
                      <span className="text-xs text-muted-foreground">
                        {client.description}
                      </span>
                    )}
                  </div>
                </TableCell>
                <TableCell data-testid="oauth-client-id">
                  <div className="rounded-md border p-1 px-2 font-mono text-xs">
                    {client.client_id}
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline">
                    {client.client_type === "confidential"
                      ? "Confidential"
                      : "Public"}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge
                    variant={
                      client.status === "ACTIVE" ? "default" : "destructive"
                    }
                    className={
                      client.status === "ACTIVE"
                        ? "border-green-600 bg-green-100 text-green-800"
                        : "border-red-600 bg-red-100 text-red-800"
                    }
                  >
                    {client.status}
                  </Badge>
                </TableCell>
                <TableCell>
                  {new Date(client.created_at).toLocaleDateString()}
                </TableCell>
                <TableCell>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        data-testid="oauth-client-actions"
                        variant="ghost"
                        size="sm"
                      >
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => handleEditClient(client)}
                        disabled={isActionPending}
                      >
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() =>
                          handleRotateWebhookSecret(client.client_id)
                        }
                        disabled={isActionPending}
                      >
                        Rotate Webhook Secret
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      {client.status === "ACTIVE" ? (
                        <DropdownMenuItem
                          onClick={() => handleSuspendClient(client.client_id)}
                          disabled={isActionPending}
                        >
                          Suspend
                        </DropdownMenuItem>
                      ) : (
                        <DropdownMenuItem
                          onClick={() => handleActivateClient(client.client_id)}
                          disabled={isActionPending}
                        >
                          Activate
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive"
                        onClick={() => handleDeleteClient(client.client_id)}
                        disabled={isActionPending}
                      >
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : (
        <div className="py-8 text-center text-muted-foreground">
          No OAuth clients registered yet. Create one to get started.
        </div>
      )}

      <Dialog
        open={webhookSecretDialogOpen}
        onOpenChange={setWebhookSecretDialogOpen}
      >
        <DialogContent className="sm:max-w-[525px]">
          <DialogHeader>
            <DialogTitle>Webhook Secret Rotated</DialogTitle>
            <DialogDescription>
              Your new webhook secret has been generated. Please copy it now as
              it will not be shown again.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>New Webhook Secret</Label>
              <div className="flex items-center space-x-2">
                <code className="flex-1 break-all rounded-md bg-secondary p-2 font-mono text-sm">
                  {newWebhookSecret}
                </code>
                <Button
                  size="icon"
                  variant="outline"
                  onClick={handleCopyWebhookSecret}
                >
                  <LuCopy className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Use this secret to verify webhook signatures (HMAC-SHA256)
              </p>
            </div>
            <p className="text-xs text-destructive">
              This secret will only be shown once. Store it securely!
            </p>
          </div>
          <DialogFooter>
            <Button onClick={() => setWebhookSecretDialogOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[525px]">
          <DialogHeader>
            <DialogTitle>Edit OAuth Client</DialogTitle>
            <DialogDescription>
              Update your OAuth client settings. Changes will take effect
              immediately.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="edit-name">Name</Label>
              <Input
                id="edit-name"
                value={editFormState.name ?? ""}
                onChange={(e) =>
                  setEditFormState((prev) => ({
                    ...prev,
                    name: e.target.value,
                  }))
                }
                placeholder="My Application"
                maxLength={100}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="edit-description">Description</Label>
              <Input
                id="edit-description"
                value={editFormState.description ?? ""}
                onChange={(e) =>
                  setEditFormState((prev) => ({
                    ...prev,
                    description: e.target.value,
                  }))
                }
                placeholder="A brief description of your application"
                maxLength={500}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="edit-redirectUris">Redirect URIs</Label>
              <Textarea
                id="edit-redirectUris"
                value={editFormState.redirect_uris?.join("\n") ?? ""}
                onChange={(e) =>
                  setEditFormState((prev) => ({
                    ...prev,
                    redirect_uris: e.target.value
                      .split(/[\n,]/)
                      .map((uri) => uri.trim())
                      .filter(Boolean),
                  }))
                }
                placeholder="https://myapp.com/callback&#10;https://localhost:3000/callback"
                rows={3}
              />
              <p className="text-xs text-muted-foreground">
                Enter one URI per line or separate with commas
              </p>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="edit-webhookDomains">Webhook Domains</Label>
              <Textarea
                id="edit-webhookDomains"
                value={editFormState.webhook_domains?.join("\n") ?? ""}
                onChange={(e) =>
                  setEditFormState((prev) => ({
                    ...prev,
                    webhook_domains: e.target.value
                      .split(/[\n,]/)
                      .map((domain) => domain.trim())
                      .filter(Boolean),
                  }))
                }
                placeholder="https://myapp.com&#10;https://api.myapp.com"
                rows={3}
              />
              <p className="text-xs text-muted-foreground">
                Domains that can receive webhook notifications
              </p>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="edit-homepageUrl">Homepage URL</Label>
              <Input
                id="edit-homepageUrl"
                type="url"
                value={editFormState.homepage_url ?? ""}
                onChange={(e) =>
                  setEditFormState((prev) => ({
                    ...prev,
                    homepage_url: e.target.value || undefined,
                  }))
                }
                placeholder="https://myapp.com"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="edit-privacyPolicyUrl">Privacy Policy URL</Label>
              <Input
                id="edit-privacyPolicyUrl"
                type="url"
                value={editFormState.privacy_policy_url ?? ""}
                onChange={(e) =>
                  setEditFormState((prev) => ({
                    ...prev,
                    privacy_policy_url: e.target.value || undefined,
                  }))
                }
                placeholder="https://myapp.com/privacy"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="edit-termsOfServiceUrl">
                Terms of Service URL
              </Label>
              <Input
                id="edit-termsOfServiceUrl"
                type="url"
                value={editFormState.terms_of_service_url ?? ""}
                onChange={(e) =>
                  setEditFormState((prev) => ({
                    ...prev,
                    terms_of_service_url: e.target.value || undefined,
                  }))
                }
                placeholder="https://myapp.com/terms"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveClient} disabled={isUpdating}>
              {isUpdating ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
