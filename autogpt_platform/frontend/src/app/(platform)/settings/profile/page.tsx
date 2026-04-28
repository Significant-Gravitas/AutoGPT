"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { LinksSection } from "./components/LinksSection/LinksSection";
import { ProfileForm } from "./components/ProfileForm/ProfileForm";
import { ProfileHeader } from "./components/ProfileHeader/ProfileHeader";
import { ProfileSkeleton } from "./components/ProfileSkeleton/ProfileSkeleton";
import { SaveBar } from "./components/SaveBar/SaveBar";
import { useProfilePage } from "./useProfilePage";

export default function SettingsProfilePage() {
  const {
    user,
    isLoading,
    isError,
    error,
    refetch,
    formState,
    patchField,
    setLink,
    addLink,
    removeLink,
    discardChanges,
    uploadAvatar,
    isUploading,
    saveProfile,
    isSaving,
    dirty,
    validation,
  } = useProfilePage();

  if (isError) {
    return (
      <ErrorCard
        context="profile"
        responseError={error ? { detail: error.detail } : undefined}
        onRetry={() => {
          void refetch();
        }}
      />
    );
  }

  if (isLoading || !user) {
    return <ProfileSkeleton />;
  }

  return (
    <div className="flex w-full flex-col gap-6">
      <div className="flex flex-col pb-2">
        <Text variant="h4" as="h1" className="leading-[28px] text-textBlack">
          Profile
        </Text>
        <Text variant="body" className="mt-4 max-w-[600px] text-zinc-700">
          Manage how you appear on the marketplace — your photo, handle, bio,
          and links.
        </Text>
      </div>

      <ProfileHeader
        avatarUrl={formState.avatar_url}
        name={formState.name}
        username={formState.username}
        errors={validation.errors}
        onChange={patchField}
        isUploading={isUploading}
        onUpload={uploadAvatar}
      />

      <ProfileForm
        formState={formState}
        errors={validation.errors}
        onChange={patchField}
      />

      <LinksSection
        links={formState.links}
        onChange={setLink}
        onAdd={addLink}
        onRemove={removeLink}
      />

      <SaveBar
        dirty={dirty}
        saving={isSaving}
        canSave={validation.valid && dirty}
        onDiscard={discardChanges}
        onSave={saveProfile}
      />
    </div>
  );
}
