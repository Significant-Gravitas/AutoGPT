"use client";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/molecules/Dialog/components/DialogPrimitives";
import { LuCopy } from "react-icons/lu";
import { Label } from "@/components/atoms/Label/Label";
import { BaseInput } from "@/components/atoms/Input/BaseInput";
import { Text } from "@/components/atoms/Text/Text";
import { Checkbox } from "@/components/atoms/Checkbox/Checkbox";
import { Button } from "@/components/atoms/Button/Button";

import { useAPIkeysModals } from "./useAPIkeysModals";
import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";

export const APIKeysModals = () => {
  const {
    isCreating,
    handleCreateKey,
    handleCopyKey,
    setIsCreateOpen,
    setIsKeyDialogOpen,
    isCreateOpen,
    isKeyDialogOpen,
    keyState,
    setKeyState,
  } = useAPIkeysModals();

  return (
    <div className="mb-4 flex justify-end">
      <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
        <DialogTrigger asChild>
          <Button variant="legacy-default">Create Key</Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New API Key</DialogTitle>
            <DialogDescription>
              Create a new AutoGPT Platform API key
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Name</Label>
              <BaseInput
                id="name"
                value={keyState.newKeyName}
                onChange={(e) =>
                  setKeyState((prev) => ({
                    ...prev,
                    newKeyName: e.target.value,
                  }))
                }
                placeholder="My AutoGPT Platform API Key"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">Description (Optional)</Label>
              <BaseInput
                id="description"
                value={keyState.newKeyDescription}
                onChange={(e) =>
                  setKeyState((prev) => ({
                    ...prev,
                    newKeyDescription: e.target.value,
                  }))
                }
                placeholder="Used for..."
              />
            </div>
            <div className="grid gap-2">
              <Label>Permissions</Label>
              <div className="max-h-48 space-y-2 overflow-y-auto">
                {Object.values(APIKeyPermission).map((permission) => (
                  <div className="flex items-center space-x-2" key={permission}>
                    <Checkbox
                      id={permission}
                      checked={keyState.selectedPermissions.includes(
                        permission,
                      )}
                      onCheckedChange={(checked: boolean) => {
                        setKeyState((prev) => ({
                          ...prev,
                          selectedPermissions: checked
                            ? [...prev.selectedPermissions, permission]
                            : prev.selectedPermissions.filter(
                                (p) => p !== permission,
                              ),
                        }));
                      }}
                    />
                    <Label htmlFor={permission}>{permission}</Label>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="legacy-outline"
              onClick={() => setIsCreateOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="legacy-default"
              onClick={handleCreateKey}
              disabled={isCreating}
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isKeyDialogOpen} onOpenChange={setIsKeyDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>AutoGPT Platform API Key Created</DialogTitle>
            <DialogDescription>
              Please copy your AutoGPT API key now. You won&apos;t be able to
              see it again!
            </DialogDescription>
          </DialogHeader>
          <div className="flex items-center space-x-2">
            <Text
              variant="body-mono-tight"
              as="code"
              tone="inherit"
              className="ph-no-capture flex-1 rounded-md bg-secondary p-2"
            >
              {keyState.newApiKey}
            </Text>
            <Button
              size="legacy-icon"
              variant="legacy-outline"
              onClick={handleCopyKey}
            >
              <LuCopy className="h-4 w-4" />
            </Button>
          </div>
          <DialogFooter>
            <Button
              variant="legacy-default"
              onClick={() => setIsKeyDialogOpen(false)}
            >
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
