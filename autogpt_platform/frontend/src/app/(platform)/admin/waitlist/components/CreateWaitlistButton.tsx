"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  usePostV2CreateWaitlist,
  getGetV2ListAllWaitlistsQueryKey,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Plus } from "@phosphor-icons/react";

export function CreateWaitlistButton() {
  const [open, setOpen] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const createWaitlistMutation = usePostV2CreateWaitlist({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Success",
            description: "Waitlist created successfully",
          });
          setOpen(false);
          setFormData({
            name: "",
            slug: "",
            subHeading: "",
            description: "",
            categories: "",
            imageUrls: "",
            videoUrl: "",
            agentOutputDemoUrl: "",
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2ListAllWaitlistsQueryKey(),
          });
        } else {
          toast({
            variant: "destructive",
            title: "Error",
            description: "Failed to create waitlist",
          });
        }
      },
      onError: (error) => {
        console.error("Error creating waitlist:", error);
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to create waitlist",
        });
      },
    },
  });

  const [formData, setFormData] = useState({
    name: "",
    slug: "",
    subHeading: "",
    description: "",
    categories: "",
    imageUrls: "",
    videoUrl: "",
    agentOutputDemoUrl: "",
  });

  function handleInputChange(id: string, value: string) {
    setFormData((prev) => ({
      ...prev,
      [id]: value,
    }));
  }

  function generateSlug(name: string) {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    createWaitlistMutation.mutate({
      data: {
        name: formData.name,
        slug: formData.slug || generateSlug(formData.name),
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
      },
    });
  }

  return (
    <>
      <Button onClick={() => setOpen(true)}>
        <Plus size={16} className="mr-2" />
        Create Waitlist
      </Button>

      <Dialog
        title="Create New Waitlist"
        controlled={{
          isOpen: open,
          set: async (isOpen) => setOpen(isOpen),
        }}
        onClose={() => setOpen(false)}
        styling={{ maxWidth: "600px" }}
      >
        <Dialog.Content>
          <p className="mb-4 text-sm text-zinc-500">
            Create a new waitlist for an upcoming agent. Users can sign up to be
            notified when it launches.
          </p>
          <form onSubmit={handleSubmit} className="flex flex-col gap-2">
            <Input
              id="name"
              label="Name"
              value={formData.name}
              onChange={(e) => handleInputChange("name", e.target.value)}
              placeholder="SEO Analysis Agent"
              required
            />

            <Input
              id="slug"
              label="Slug"
              value={formData.slug}
              onChange={(e) => handleInputChange("slug", e.target.value)}
              placeholder="seo-analysis-agent (auto-generated if empty)"
            />

            <Input
              id="subHeading"
              label="Subheading"
              value={formData.subHeading}
              onChange={(e) => handleInputChange("subHeading", e.target.value)}
              placeholder="Analyze your website's SEO in minutes"
              required
            />

            <Input
              id="description"
              label="Description"
              type="textarea"
              value={formData.description}
              onChange={(e) => handleInputChange("description", e.target.value)}
              placeholder="Detailed description of what this agent does..."
              rows={4}
              required
            />

            <Input
              id="categories"
              label="Categories (comma-separated)"
              value={formData.categories}
              onChange={(e) => handleInputChange("categories", e.target.value)}
              placeholder="SEO, Marketing, Analysis"
            />

            <Input
              id="imageUrls"
              label="Image URLs (comma-separated)"
              value={formData.imageUrls}
              onChange={(e) => handleInputChange("imageUrls", e.target.value)}
              placeholder="https://example.com/image1.jpg, https://example.com/image2.jpg"
            />

            <Input
              id="videoUrl"
              label="Video URL (optional)"
              value={formData.videoUrl}
              onChange={(e) => handleInputChange("videoUrl", e.target.value)}
              placeholder="https://youtube.com/watch?v=..."
            />

            <Input
              id="agentOutputDemoUrl"
              label="Output Demo URL (optional)"
              value={formData.agentOutputDemoUrl}
              onChange={(e) =>
                handleInputChange("agentOutputDemoUrl", e.target.value)
              }
              placeholder="https://example.com/demo-output.mp4"
            />

            <Dialog.Footer>
              <Button
                type="button"
                variant="secondary"
                onClick={() => setOpen(false)}
              >
                Cancel
              </Button>
              <Button type="submit" loading={createWaitlistMutation.isPending}>
                Create Waitlist
              </Button>
            </Dialog.Footer>
          </form>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
