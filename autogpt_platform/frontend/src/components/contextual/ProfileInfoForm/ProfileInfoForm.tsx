"use client";

import Image from "next/image";
import { UserIcon } from "@hugeicons/core-free-icons";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Separator } from "@/components/ui/separator";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { useProfileInfoForm } from "./useProfileInfoForm";

interface Props {
  profile: ProfileDetails;
}

export function ProfileInfoForm({ profile }: Props) {
  const {
    profileData,
    setProfileData,
    isSubmitting,
    submitForm,
    handleImageUpload,
  } = useProfileInfoForm(profile);

  return (
    <div className="w-full min-w-[800px] px-4 sm:px-8">
      <Text
        variant="h2"
        as="h1"
        tone="primary"
        data-testid="profile-info-form-title"
        className="mb-6 sm:mb-8"
      >
        Profile
      </Text>

      <div className="mb-8 sm:mb-12">
        <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row sm:items-start">
          <div className="relative h-[130px] w-[130px] rounded-full bg-zinc-200">
            {profileData.avatar_url ? (
              <Image
                src={profileData.avatar_url}
                unoptimized={isLocalStoreMediaUrl(profileData.avatar_url)}
                alt="Profile"
                fill
                className="rounded-full"
              />
            ) : (
              <Icon
                icon={UserIcon}
                size={72}
                aria-label="Person Fill Icon"
                className="absolute inset-0 m-auto text-muted-foreground"
              />
            )}
          </div>
          <label className="mt-11 inline-flex h-[43px] items-center justify-center rounded-full bg-black px-6 py-2 text-sm text-white transition-colors hover:bg-zinc-900">
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (file) {
                  await handleImageUpload(file);
                }
              }}
            />
            Edit photo
          </label>
        </div>

        <form className="space-y-4 sm:space-y-6" onSubmit={submitForm}>
          <Input
            type="text"
            id="displayName"
            name="displayName"
            label="Display name"
            labelVariant="large"
            labelClassName="text-zinc-700"
            wrapperClassName="mb-0"
            data-testid="profile-info-form-display-name"
            defaultValue={profileData.name}
            placeholder="Enter your display name"
            onChange={(e) => {
              const newProfileData = {
                ...profileData,
                name: e.target.value,
              };
              setProfileData(newProfileData);
            }}
          />

          <Input
            type="text"
            id="handle"
            name="handle"
            label="Handle"
            labelVariant="large"
            labelClassName="text-zinc-700"
            wrapperClassName="mb-0"
            defaultValue={profileData.username}
            placeholder="@username"
            onChange={(e) => {
              const newProfileData = {
                ...profileData,
                username: e.target.value,
              };
              setProfileData(newProfileData);
            }}
          />

          <Input
            type="textarea"
            id="bio"
            name="bio"
            label="Bio"
            labelVariant="large"
            labelClassName="text-zinc-700"
            wrapperClassName="mb-0"
            rows={8}
            value={profileData.description}
            placeholder="Tell us about yourself..."
            className="resize-none"
            onChange={(e) => {
              const newProfileData = {
                ...profileData,
                description: e.target.value,
              };
              setProfileData(newProfileData);
            }}
          />

          <section className="mb-8">
            <Text variant="large" as="h2" tone="secondary" className="mb-4">
              Your links
            </Text>
            <Text variant="large" tone="secondary" className="mb-6">
              You can display up to 5 links on your profile
            </Text>

            <div className="space-y-4 sm:space-y-6">
              {[1, 2, 3, 4, 5].map((linkNum) => {
                const link = profileData.links[linkNum - 1];
                return (
                  <Input
                    key={linkNum}
                    type="text"
                    id={`link${linkNum}`}
                    name={`link${linkNum}`}
                    label={`Link ${linkNum}`}
                    labelVariant="large"
                    labelClassName="text-zinc-700"
                    wrapperClassName="mb-0"
                    placeholder="https://"
                    defaultValue={link || ""}
                    onChange={(e) => {
                      const newLinks = [...profileData.links];
                      newLinks[linkNum - 1] = e.target.value;
                      const newProfileData = {
                        ...profileData,
                        links: newLinks,
                      };
                      setProfileData(newProfileData);
                    }}
                  />
                );
              })}
            </div>
          </section>

          <Separator />

          <div className="flex h-[50px] items-center justify-end gap-3 py-8">
            <Button
              type="submit"
              disabled={isSubmitting}
              loading={isSubmitting}
              className="h-[50px] px-6 py-3 text-base"
            >
              {isSubmitting ? "Saving..." : "Save changes"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
