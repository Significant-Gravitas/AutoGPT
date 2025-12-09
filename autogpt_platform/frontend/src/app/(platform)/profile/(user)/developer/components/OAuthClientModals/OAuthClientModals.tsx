"use client";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/__legacy__/ui/dialog";
import { LuCopy } from "react-icons/lu";
import { Label } from "@/components/__legacy__/ui/label";
import { Input } from "@/components/__legacy__/ui/input";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { Button } from "@/components/__legacy__/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";

import { useOAuthClientModals } from "./useOAuthClientModals";

export function OAuthClientModals() {
  const {
    isCreateOpen,
    setIsCreateOpen,
    isSecretDialogOpen,
    setIsSecretDialogOpen,
    formState,
    setFormState,
    newClientSecret,
    isCreating,
    handleCreateClient,
    handleCopyClientId,
    handleCopyClientSecret,
    resetForm,
  } = useOAuthClientModals();

  return (
    <div className="mb-4 flex justify-end">
      <Dialog
        open={isCreateOpen}
        onOpenChange={(open) => {
          setIsCreateOpen(open);
          if (!open) resetForm();
        }}
      >
        <DialogTrigger asChild>
          <Button>Register OAuth Client</Button>
        </DialogTrigger>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[525px]">
          <DialogHeader>
            <DialogTitle>Register New OAuth Client</DialogTitle>
            <DialogDescription>
              Register a new OAuth client to integrate with the AutoGPT
              Platform. For confidential clients, the client secret will only be
              shown once.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">
                Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="name"
                value={formState.name}
                onChange={(e) =>
                  setFormState((prev) => ({
                    ...prev,
                    name: e.target.value,
                  }))
                }
                placeholder="My Application"
                maxLength={100}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                value={formState.description}
                onChange={(e) =>
                  setFormState((prev) => ({
                    ...prev,
                    description: e.target.value,
                  }))
                }
                placeholder="A brief description of your application"
                maxLength={500}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="redirectUris">
                Redirect URIs <span className="text-destructive">*</span>
              </Label>
              <Textarea
                id="redirectUris"
                value={formState.redirectUris}
                onChange={(e) =>
                  setFormState((prev) => ({
                    ...prev,
                    redirectUris: e.target.value,
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
              <Label htmlFor="clientType">Client Type</Label>
              <Select
                value={formState.clientType}
                onValueChange={(value: "public" | "confidential") =>
                  setFormState((prev) => ({
                    ...prev,
                    clientType: value,
                  }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select client type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="public">
                    Public (SPA, Mobile apps)
                  </SelectItem>
                  <SelectItem value="confidential">
                    Confidential (Server-side apps)
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Public clients cannot securely store secrets. Confidential
                clients receive a client secret.
              </p>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="homepageUrl">Homepage URL</Label>
              <Input
                id="homepageUrl"
                type="url"
                value={formState.homepageUrl}
                onChange={(e) =>
                  setFormState((prev) => ({
                    ...prev,
                    homepageUrl: e.target.value,
                  }))
                }
                placeholder="https://myapp.com"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="privacyPolicyUrl">Privacy Policy URL</Label>
              <Input
                id="privacyPolicyUrl"
                type="url"
                value={formState.privacyPolicyUrl}
                onChange={(e) =>
                  setFormState((prev) => ({
                    ...prev,
                    privacyPolicyUrl: e.target.value,
                  }))
                }
                placeholder="https://myapp.com/privacy"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="termsOfServiceUrl">Terms of Service URL</Label>
              <Input
                id="termsOfServiceUrl"
                type="url"
                value={formState.termsOfServiceUrl}
                onChange={(e) =>
                  setFormState((prev) => ({
                    ...prev,
                    termsOfServiceUrl: e.target.value,
                  }))
                }
                placeholder="https://myapp.com/terms"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsCreateOpen(false);
                resetForm();
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateClient} disabled={isCreating}>
              {isCreating ? "Creating..." : "Create Client"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isSecretDialogOpen} onOpenChange={setIsSecretDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>OAuth Client Created</DialogTitle>
            <DialogDescription>
              Please copy your client credentials now. The client secret will
              not be shown again!
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Client ID</Label>
              <div className="flex items-center space-x-2">
                <code className="flex-1 rounded-md bg-secondary p-2 font-mono text-sm">
                  {newClientSecret?.client_id}
                </code>
                <Button size="icon" variant="outline" onClick={handleCopyClientId}>
                  <LuCopy className="h-4 w-4" />
                </Button>
              </div>
            </div>
            {newClientSecret?.client_secret && (
              <div className="space-y-2">
                <Label>Client Secret</Label>
                <div className="flex items-center space-x-2">
                  <code className="flex-1 break-all rounded-md bg-secondary p-2 font-mono text-sm">
                    {newClientSecret.client_secret}
                  </code>
                  <Button
                    size="icon"
                    variant="outline"
                    onClick={handleCopyClientSecret}
                  >
                    <LuCopy className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-xs text-destructive">
                  This secret will only be shown once. Store it securely!
                </p>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button onClick={() => setIsSecretDialogOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
