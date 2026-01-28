"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePutV2UpdateWaitlist } from "@/app/api/__generated__/endpoints/admin/admin";
import type { WaitlistAdminResponse } from "@/app/api/__generated__/models/waitlistAdminResponse";
import type { WaitlistUpdateRequest } from "@/app/api/__generated__/models/waitlistUpdateRequest";
import { WaitlistExternalStatus } from "@/app/api/__generated__/models/waitlistExternalStatus";

type EditWaitlistDialogProps = {
  waitlist: WaitlistAdminResponse;
  onClose: () => void;
  onSave: () => void;
};

const STATUS_OPTIONS = [
  { value: WaitlistExternalStatus.NOT_STARTED, label: "Not Started" },
  { value: WaitlistExternalStatus.WORK_IN_PROGRESS, label: "Work In Progress" },
  { value: WaitlistExternalStatus.DONE, label: "Done" },
  { value: WaitlistExternalStatus.CANCELED, label: "Canceled" },
];

export function EditWaitlistDialog({
  waitlist,
  onClose,
  onSave,
}: EditWaitlistDialogProps) {
  const { toast } = useToast();
  const updateWaitlistMutation = usePutV2UpdateWaitlist();

  const [formData, setFormData] = useState({
    name: waitlist.name,
    slug: waitlist.slug,
    subHeading: waitlist.subHeading,
    description: waitlist.description,
    categories: waitlist.categories.join(", "),
    imageUrls: waitlist.imageUrls.join(", "),
    videoUrl: waitlist.videoUrl || "",
    agentOutputDemoUrl: waitlist.agentOutputDemoUrl || "",
    status: waitlist.status,
    storeListingId: waitlist.storeListingId || "",
  });

  function handleInputChange(id: string, value: string) {
    setFormData((prev) => ({
      ...prev,
      [id]: value,
    }));
  }

  function handleStatusChange(value: string) {
    setFormData((prev) => ({
      ...prev,
      status: value as WaitlistExternalStatus,
    }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    const updateData: WaitlistUpdateRequest = {
      name: formData.name,
      slug: formData.slug,
      subHeading: formData.subHeading,
      description: formData.description,
      categories: formData.categories
        ? formData.categories.split(",").map((c) => c.trim())
        : [],
      imageUrls: formData.imageUrls
        ? formData.imageUrls.split(",").map((u) => u.trim())
        : [],
      videoUrl: formData.videoUrl || null,
      agentOutputDemoUrl: formData.agentOutputDemoUrl || null,
      status: formData.status,
      storeListingId: formData.storeListingId || null,
    };

    updateWaitlistMutation.mutate(
      { waitlistId: waitlist.id, data: updateData },
      {
        onSuccess: (response) => {
          if (response.status === 200) {
            toast({
              title: "Success",
              description: "Waitlist updated successfully",
            });
            onSave();
          } else {
            toast({
              variant: "destructive",
              title: "Error",
              description: "Failed to update waitlist",
            });
          }
        },
        onError: () => {
          toast({
            variant: "destructive",
            title: "Error",
            description: "Failed to update waitlist",
          });
        },
      },
    );
  }

  return (
    <Dialog
      title="Edit Waitlist"
      controlled={{
        isOpen: true,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "600px" }}
    >
      <Dialog.Content>
        <p className="mb-4 text-sm text-zinc-500">
          Update the waitlist details. Changes will be reflected immediately.
        </p>
        <form onSubmit={handleSubmit} className="flex flex-col gap-2">
          <Input
            id="name"
            label="Name"
            value={formData.name}
            onChange={(e) => handleInputChange("name", e.target.value)}
            required
          />

          <Input
            id="slug"
            label="Slug"
            value={formData.slug}
            onChange={(e) => handleInputChange("slug", e.target.value)}
          />

          <Input
            id="subHeading"
            label="Subheading"
            value={formData.subHeading}
            onChange={(e) => handleInputChange("subHeading", e.target.value)}
            required
          />

          <Input
            id="description"
            label="Description"
            type="textarea"
            value={formData.description}
            onChange={(e) => handleInputChange("description", e.target.value)}
            rows={4}
            required
          />

          <Select
            id="status"
            label="Status"
            value={formData.status}
            onValueChange={handleStatusChange}
            options={STATUS_OPTIONS}
          />

          <Input
            id="categories"
            label="Categories (comma-separated)"
            value={formData.categories}
            onChange={(e) => handleInputChange("categories", e.target.value)}
          />

          <Input
            id="imageUrls"
            label="Image URLs (comma-separated)"
            value={formData.imageUrls}
            onChange={(e) => handleInputChange("imageUrls", e.target.value)}
          />

          <Input
            id="videoUrl"
            label="Video URL"
            value={formData.videoUrl}
            onChange={(e) => handleInputChange("videoUrl", e.target.value)}
          />

          <Input
            id="agentOutputDemoUrl"
            label="Output Demo URL"
            value={formData.agentOutputDemoUrl}
            onChange={(e) =>
              handleInputChange("agentOutputDemoUrl", e.target.value)
            }
          />

          <Input
            id="storeListingId"
            label="Store Listing ID (for linking)"
            value={formData.storeListingId}
            onChange={(e) =>
              handleInputChange("storeListingId", e.target.value)
            }
            placeholder="Leave empty if not linked"
          />

          <Dialog.Footer>
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" loading={updateWaitlistMutation.isPending}>
              Save Changes
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
