"use client";

import * as React from "react";
import { useState } from "react";

import Image from "next/image";

import { Button } from "./Button";

import { IconPersonFill } from "@/components/ui/icons";

import AutoGPTServerAPI from "@/lib/autogpt-server-api/client";
import { CreatorDetails, ProfileDetails } from "@/lib/autogpt-server-api/types";
import { createClient } from "@/lib/supabase/client";
import { Separator } from "@/components/ui/separator";

export const ProfileInfoForm = ({ profile }: { profile: CreatorDetails }) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [profileData, setProfileData] = useState(profile);

  const supabase = createClient();

  const api = new AutoGPTServerAPI(
    process.env.NEXT_PUBLIC_AGPT_SERVER_URL,
    process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL,
    supabase,
  );

  const submitForm = async () => {
    try {
      setIsSubmitting(true);

      const updatedProfile = {
        name: profileData.name,
        username: profileData.username,
        description: profileData.description,
        links: profileData.links,
        avatar_url: profileData.avatar_url,
      };

      if (!isSubmitting) {
        const returnedProfile = await api.updateStoreProfile(
          updatedProfile as ProfileDetails,
        );
        setProfileData(returnedProfile as CreatorDetails);
      }
    } catch (error) {
      console.error("Error updating profile:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleImageUpload = async (file: File) => {
    try {
      // Create FormData and append file
      const formData = new FormData();
      formData.append("file", file);

      console.log(formData);

      // Get auth token
      if (!supabase) {
        throw new Error("Supabase client not initialized");
      }

      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;

      if (!token) {
        throw new Error("No authentication token found");
      }

      // Make upload request
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_AGPT_SERVER_URL}/store/submissions/media`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      // Get media URL from response
      const mediaUrl = await response.json();

      // Update profile with new avatar URL
      const updatedProfile = {
        ...profileData,
        avatar_url: mediaUrl,
      };

      const returnedProfile = await api.updateStoreProfile(
        updatedProfile as ProfileDetails,
      );
      setProfileData(returnedProfile as CreatorDetails);
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  return (
    <div className="w-full min-w-[600px] px-4 sm:px-8">
      <h1 className="mb-6 font-['Poppins'] text-[28px] font-medium text-neutral-900 dark:text-neutral-100 sm:mb-8 sm:text-[35px]">
        Profile
      </h1>

      <div className="mb-8 sm:mb-12">
        <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row sm:items-start">
          <div className="relative h-[130px] w-[130px] rounded-full bg-[#d9d9d9] dark:bg-[#333333]">
            {profileData.avatar_url ? (
              <Image
                src={profileData.avatar_url}
                alt="Profile"
                fill
                className="rounded-full"
              />
            ) : (
              <IconPersonFill className="absolute left-[30px] top-[24px] h-[77.80px] w-[70.63px] text-[#7e7e7e] dark:text-[#999999]" />
            )}
          </div>
          <label className="mt-11 inline-flex h-[43px] items-center justify-center rounded-[22px] border border-slate-900 bg-slate-900 px-4 py-2 font-['Geist'] text-sm font-medium leading-normal text-slate-50 transition-colors hover:bg-white hover:text-slate-900 dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100">
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
            <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
              Display name
            </label>
            <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700 dark:bg-slate-800">
              <input
                type="text"
                name="displayName"
                defaultValue={profileData.name}
                placeholder="Enter your display name"
                className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
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
            <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
              Handle
            </label>
            <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700 dark:bg-slate-800">
              <input
                type="text"
                name="handle"
                defaultValue={profileData.username}
                placeholder="@username"
                className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
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
            <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
              Bio
            </label>
            <div className="h-[220px] rounded-2xl border border-slate-200 py-2.5 pl-4 pr-4 dark:border-slate-700 dark:bg-slate-800">
              <textarea
                name="bio"
                defaultValue={profileData.description}
                placeholder="Tell us about yourself..."
                className="h-full w-full resize-none border-none bg-transparent font-['Geist'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
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
            <h2 className="mb-4 font-['Poppins'] text-lg font-semibold leading-7 text-neutral-500 dark:text-neutral-400">
              Your links
            </h2>
            <p className="mb-6 font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
              You can display up to 5 links on your profile
            </p>

            <div className="space-y-4 sm:space-y-6">
              {[1, 2, 3, 4, 5].map((linkNum) => {
                const link = profileData.links[linkNum - 1];
                return (
                  <div key={linkNum} className="w-full">
                    <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
                      Link {linkNum}
                    </label>
                    <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700 dark:bg-slate-800">
                      <input
                        type="text"
                        name={`link${linkNum}`}
                        placeholder="https://"
                        defaultValue={link || ""}
                        className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
                        onChange={(e) => {
                          const newProfileData = {
                            ...profileData,
                            links: profileData.links.map((link, index) =>
                              index === linkNum - 1 ? e.target.value : link,
                            ),
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
              type="button"
              variant="secondary"
              className="h-[50px] rounded-[35px] bg-neutral-200 px-6 py-3 font-['Geist'] text-base font-medium text-neutral-800 transition-colors hover:bg-neutral-300 dark:border-neutral-700 dark:bg-neutral-700 dark:text-neutral-200 dark:hover:border-neutral-600 dark:hover:bg-neutral-600"
              onClick={() => {
                setProfileData(profile);
              }}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="default"
              disabled={isSubmitting}
              className="h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 font-['Geist'] text-base font-medium text-white transition-colors hover:bg-neutral-900 dark:bg-neutral-200 dark:text-neutral-900 dark:hover:bg-neutral-100"
              onClick={submitForm}
            >
              {isSubmitting ? "Saving..." : "Save changes"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
