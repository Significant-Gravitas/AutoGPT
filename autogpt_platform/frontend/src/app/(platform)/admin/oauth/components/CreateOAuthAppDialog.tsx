"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { Button } from "@/components/__legacy__/ui/button";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import { Copy, Eye, EyeOff, Plus, X } from "lucide-react";
import { createOAuthApp } from "../actions";
import type { OAuthApplicationCreationResult } from "@/lib/autogpt-server-api/types";

const AVAILABLE_SCOPES = [
  { value: "EXECUTE_GRAPH", label: "Execute Graph", description: "Run agent graphs" },
  { value: "READ_GRAPH", label: "Read Graph", description: "Read agent graphs" },
  { value: "EXECUTE_BLOCK", label: "Execute Block", description: "Execute individual blocks" },
  { value: "READ_BLOCK", label: "Read Block", description: "Read block definitions" },
  { value: "READ_STORE", label: "Read Store", description: "Access the store" },
  { value: "USE_TOOLS", label: "Use Tools", description: "Use available tools" },
  { value: "MANAGE_INTEGRATIONS", label: "Manage Integrations", description: "Manage integrations" },
  { value: "READ_INTEGRATIONS", label: "Read Integrations", description: "Read integrations" },
  { value: "DELETE_INTEGRATIONS", label: "Delete Integrations", description: "Delete integrations" },
];

interface CreateOAuthAppDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateOAuthAppDialog({ open, onOpenChange }: CreateOAuthAppDialogProps) {
  const [step, setStep] = useState<"form" | "success">("form");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OAuthApplicationCreationResult | null>(null);
  const [isSecretVisible, setIsSecretVisible] = useState(false);

  // Form state
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [ownerId, setOwnerId] = useState("");
  const [redirectUris, setRedirectUris] = useState<string[]>([""]);
  const [selectedScopes, setSelectedScopes] = useState<string[]>([]);

  const resetForm = () => {
    setStep("form");
    setIsLoading(false);
    setError(null);
    setResult(null);
    setIsSecretVisible(false);
    setName("");
    setDescription("");
    setOwnerId("");
    setRedirectUris([""]);
    setSelectedScopes([]);
  };

  const handleClose = () => {
    resetForm();
    onOpenChange(false);
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
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    // Validation
    const validUris = redirectUris.filter((uri) => uri.trim());
    if (!name.trim()) {
      setError("Name is required");
      setIsLoading(false);
      return;
    }
    if (!ownerId.trim()) {
      setError("Owner ID is required");
      setIsLoading(false);
      return;
    }
    if (validUris.length === 0) {
      setError("At least one redirect URI is required");
      setIsLoading(false);
      return;
    }
    if (selectedScopes.length === 0) {
      setError("At least one scope is required");
      setIsLoading(false);
      return;
    }

    try {
      const creationResult = await createOAuthApp({
        name: name.trim(),
        description: description.trim() || undefined,
        owner_id: ownerId.trim(),
        redirect_uris: validUris,
        scopes: selectedScopes,
      });
      setResult(creationResult);
      setStep("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create OAuth application");
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        {step === "form" ? (
          <>
            <DialogHeader>
              <DialogTitle>Create OAuth Application</DialogTitle>
              <DialogDescription>
                Create a new OAuth application for third-party integrations.
              </DialogDescription>
            </DialogHeader>

            <form onSubmit={handleSubmit} className="space-y-6">
              {error && (
                <div className="p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
                  {error}
                </div>
              )}

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Application Name *</Label>
                    <Input
                      id="name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="My Application"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="owner">Owner User ID *</Label>
                    <Input
                      id="owner"
                      value={ownerId}
                      onChange={(e) => setOwnerId(e.target.value)}
                      placeholder="User UUID"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Describe what this application does..."
                    rows={2}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Redirect URIs *</Label>
                    <Button type="button" variant="ghost" size="sm" onClick={addRedirectUri}>
                      <Plus className="mr-1 h-3 w-3" />
                      Add URI
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {redirectUris.map((uri, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <Input
                          value={uri}
                          onChange={(e) => updateRedirectUri(index, e.target.value)}
                          placeholder="https://example.com/callback"
                        />
                        {redirectUris.length > 1 && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => removeRedirectUri(index)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Scopes *</Label>
                  <div className="grid grid-cols-2 gap-2 p-4 border rounded-md bg-gray-50">
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
                            className="text-sm font-medium cursor-pointer"
                          >
                            {scope.label}
                          </label>
                          <p className="text-xs text-gray-500">{scope.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <DialogFooter>
                <Button type="button" variant="secondary" onClick={handleClose}>
                  Cancel
                </Button>
                <Button type="submit" disabled={isLoading}>
                  {isLoading ? "Creating..." : "Create Application"}
                </Button>
              </DialogFooter>
            </form>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>OAuth Application Created</DialogTitle>
              <DialogDescription>
                Your OAuth application has been created successfully. Save the client secret now -
                it will only be shown once!
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              <div className="p-4 bg-green-50 border border-green-200 rounded-md">
                <h4 className="font-medium text-green-800 mb-2">
                  {result?.application.name}
                </h4>
                <p className="text-sm text-green-700">
                  Application created successfully with ID: {result?.application.id}
                </p>
              </div>

              <div className="space-y-3">
                <div className="space-y-1">
                  <Label className="text-sm text-gray-500">Client ID</Label>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-sm bg-gray-100 px-3 py-2 rounded break-all">
                      {result?.application.client_id}
                    </code>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(result?.application.client_id || "")}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-md space-y-2">
                  <Label className="text-sm font-medium text-yellow-800">
                    Client Secret (Save this now!)
                  </Label>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-sm bg-white px-3 py-2 rounded border break-all">
                      {isSecretVisible
                        ? result?.client_secret_plaintext
                        : "••••••••••••••••••••••••••••••••"}
                    </code>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsSecretVisible(!isSecretVisible)}
                    >
                      {isSecretVisible ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(result?.client_secret_plaintext || "")}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-xs text-yellow-700">
                    This secret will only be shown once. Store it securely!
                  </p>
                </div>
              </div>
            </div>

            <DialogFooter>
              <Button onClick={handleClose}>Done</Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
