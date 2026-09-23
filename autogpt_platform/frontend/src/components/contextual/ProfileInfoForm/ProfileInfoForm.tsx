"use client";

import Image from "next/image";
import { UserIcon } from "@phosphor-icons/react";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { Button } from "@/components/atoms/Button/Button";
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
      <h1
        data-testid="profile-info-form-title"
        className="font-circular mb-6 text-[28px] font-normal text-neutral-900 dark:text-neutral-100 sm:mb-8 sm:text-[35px]"
      >
        Profile
      </h1>

      <div className="mb-8 sm:mb-12">
        <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row sm:items-start">
          <div className="relative h-[130px] w-[130px] rounded-full bg-[#d9d9d9] dark:bg-[#333333]">
            {profileData.avatar_url ? (
              <Image
                src={profileData.avatar_url}
                unoptimized={isLocalStoreMediaUrl(profileData.avatar_url)}
                alt="Profile"
                fill
                className="rounded-full"
              />
            ) : (
              <UserIcon
                weight="fill"
                aria-label="Person Fill Icon"
                className="absolute left-[30px] top-[24px] h-[77.80px] w-[70.63px] text-[#7e7e7e] dark:text-[#999999]"
              />
            )}
          </div>
          <label className="font-circular mt-11 inline-flex h-[43px] items-center justify-center rounded-[22px] bg-[#15171A] px-6 py-2 text-sm font-normal text-white transition-colors hover:bg-[#2D2F34] dark:bg-white dark:text-[#15171A] dark:hover:bg-[#E5E5E5]">
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
          <div className="w-full">
            <label
              htmlFor="displayName"
              className="font-circular mb-1.5 block text-base font-normal leading-tight text-neutral-700 dark:text-neutral-300"
            >
              Display name
            </label>
            <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700 dark:bg-slate-800">
              <input
                type="text"
                id="displayName"
                name="displayName"
                data-testid="profile-info-form-display-name"
                defaultValue={profileData.name}
                placeholder="Enter your display name"
                className="font-circular w-full border-none bg-transparent text-base font-normal text-neutral-900 placeholder:text-neutral-400 focus:outline-none dark:text-white dark:placeholder:text-neutral-500"
                onChange={(e) => {
                  const newProfileData = {
                    ...profileData,
                    name: e.target.value,
                  };
                  setProfileData(newProfileData);
                }}
              />
            </div>
          </div>

          <div className="w-full">
            <label
              htmlFor="handle"
              className="font-circular mb-1.5 block text-base font-normal leading-tight text-neutral-700 dark:text-neutral-300"
            >
              Handle
            </label>
            <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700 dark:bg-slate-800">
              <input
                type="text"
                name="handle"
                id="handle"
                defaultValue={profileData.username}
                placeholder="@username"
                className="font-circular w-full border-none bg-transparent text-base font-normal text-neutral-900 placeholder:text-neutral-400 focus:outline-none dark:text-white dark:placeholder:text-neutral-500"
                onChange={(e) => {
                  const newProfileData = {
                    ...profileData,
                    username: e.target.value,
                  };
                  setProfileData(newProfileData);
                }}
              />
            </div>
          </div>

          <div className="w-full">
            <label
              htmlFor="bio"
              className="font-circular mb-1.5 block text-base font-normal leading-tight text-neutral-700 dark:text-neutral-300"
            >
              Bio
            </label>
            <div className="h-[220px] rounded-2xl border border-slate-200 py-2.5 pl-4 pr-4 dark:border-slate-700 dark:bg-slate-800">
              <textarea
                name="bio"
                id="bio"
                defaultValue={profileData.description}
                placeholder="Tell us about yourself..."
                className="font-circular h-full w-full resize-none border-none bg-transparent text-base font-normal text-neutral-900 placeholder:text-neutral-400 focus:outline-none dark:text-white dark:placeholder:text-neutral-500"
                onChange={(e) => {
                  const newProfileData = {
                    ...profileData,
                    description: e.target.value,
                  };
                  setProfileData(newProfileData);
                }}
              />
            </div>
          </div>

          <section className="mb-8">
            <h2 className="font-circular mb-4 text-lg font-normal leading-7 text-neutral-700 dark:text-neutral-300">
              Your links
            </h2>
            <p className="font-circular mb-6 text-base font-normal leading-tight text-neutral-600 dark:text-neutral-400">
              You can display up to 5 links on your profile
            </p>

            <div className="space-y-4 sm:space-y-6">
              {[1, 2, 3, 4, 5].map((linkNum) => {
                const link = profileData.links[linkNum - 1];
                return (
                  <div key={linkNum} className="w-full">
                    <label
                      htmlFor={`link${linkNum}`}
                      className="font-circular mb-1.5 block text-base font-normal leading-tight text-neutral-700 dark:text-neutral-300"
                    >
                      Link {linkNum}
                    </label>
                    <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700 dark:bg-slate-800">
                      <input
                        type="text"
                        name={`link${linkNum}`}
                        id={`link${linkNum}`}
                        placeholder="https://"
                        defaultValue={link || ""}
                        className="font-circular w-full border-none bg-transparent text-base font-normal text-neutral-900 placeholder:text-neutral-400 focus:outline-none dark:text-white dark:placeholder:text-neutral-500"
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
                    </div>
                  </div>
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
              className="font-circular h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 text-base font-medium text-white transition-colors hover:bg-neutral-900 dark:bg-neutral-200 dark:text-neutral-900 dark:hover:bg-neutral-100"
            >
              {isSubmitting ? "Saving..." : "Save changes"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
