"use client";

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
    <div className="flex flex-col gap-6 pb-28">
      <ProfileHeader
        avatarUrl={formState.avatar_url}
        name={formState.name}
        email={user.email}
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
        visible={dirty}
        saving={isSaving}
        canSave={validation.valid && dirty}
        onDiscard={discardChanges}
        onSave={saveProfile}
      />
    </div>
  );
}
