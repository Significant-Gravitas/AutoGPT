"use client";

import * as React from "react";
import { useState } from "react";

import Image from "next/image";

import { Button } from "./Button";
import { IconPersonFill } from "@/components/ui/icons";
import { CreatorDetails, ProfileDetails } from "@/lib/autogpt-server-api/types";
import { Separator } from "@/components/ui/separator";
import useSupabase from "@/hooks/useSupabase";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Textarea } from "../ui/textarea";

export const ProfileInfoForm = ({ profile }: { profile: CreatorDetails }) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [profileData, setProfileData] = useState(profile);
  const { supabase } = useSupabase();
  const api = useBackendAPI();

  const submitForm = async () => {
    try {
      setIsSubmitting(true);

      const updatedProfile = {
        name: profileData.name,
        username: profileData.username,
        description: profileData.description,
        links: profileData.links.filter((link) => link), // Filter out empty links
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
    <div className="md:min-w-md w-full px-4 sm:px-8">
      <h1 className="mb-6 font-poppins text-4xl font-medium text-neutral-900">
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
          <Label className="mt-11 inline-flex h-[43px] items-center justify-center rounded-[22px] bg-[#15171A] px-6 py-2 font-sans text-sm font-normal text-white transition-colors hover:cursor-pointer hover:bg-[#2D2F34] dark:bg-white dark:text-[#15171A] dark:hover:bg-[#E5E5E5]">
            <Input
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
          </Label>
        </div>

        <form className="space-y-8" onSubmit={submitForm}>
          {/* Top section */}
          <section className="space-y-6">
            <div className="w-full space-y-1.5">
              <Label className="block font-sans text-base font-medium text-[#020617]">
                Display name
              </Label>
              <Input
                type="text"
                name="displayName"
                defaultValue={profileData.name}
                placeholder="Enter your display name"
                className="h-11 w-full rounded-full border border-[#E2E8F0] px-4 py-2.5 font-inter text-base font-normal text-[#7e7e7e] outline-none"
                onChange={(e) => {
                  const newProfileData = {
                    ...profileData,
                    name: e.target.value,
                  };
                  setProfileData(newProfileData);
                }}
              />
            </div>

            <div className="w-full space-y-1.5">
              <Label className="block font-sans text-base font-medium text-[#020617]">
                Handle
              </Label>
              <Input
                type="text"
                name="handle"
                defaultValue={profileData.username}
                placeholder="@username"
                className="h-11 w-full rounded-full border border-[#E2E8F0] px-4 py-2.5 font-inter text-base font-normal text-[#7e7e7e] outline-none"
                onChange={(e) => {
                  const newProfileData = {
                    ...profileData,
                    username: e.target.value,
                  };
                  setProfileData(newProfileData);
                }}
              />
            </div>

            <div className="w-full space-y-1.5">
              <Label className="block font-sans text-base font-medium text-[#020617]">
                Bio
              </Label>
              <Textarea
                name="bio"
                defaultValue={profileData.description}
                placeholder="Tell us about yourself..."
                className="min-h-56 w-full resize-none rounded-[1rem] border border-[#E2E8F0] px-4 py-2.5 font-inter text-base font-normal text-[#7e7e7e] outline-none"
                onChange={(e) => {
                  const newProfileData = {
                    ...profileData,
                    description: e.target.value,
                  };
                  setProfileData(newProfileData);
                }}
              />
            </div>
          </section>

          <Separator className="bg-neutral-300" />

          {/* mid section */}
          <section className="mb-8 space-y-6">
            <h2 className="font-poppins text-lg font-semibold text-neutral-500">
              Your links
            </h2>
            <p className="font-sans text-base font-medium text-[#020617]">
              You can display up to 5 links on your profile
            </p>

            <div className="space-y-4 sm:space-y-6">
              {[1, 2, 3, 4, 5].map((linkNum) => {
                const link = profileData.links[linkNum - 1];
                return (
                  <div key={linkNum} className="w-full space-y-1.5">
                    <Label className="block font-sans text-base font-medium text-[#020617]">
                      Link {linkNum}
                    </Label>
                    <Input
                      type="text"
                      name={`link${linkNum}`}
                      placeholder="https://"
                      defaultValue={link || ""}
                      className="h-11 w-full rounded-full border border-[#E2E8F0] px-4 py-2.5 font-inter text-base font-normal text-[#7e7e7e] outline-none"
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
                );
              })}
            </div>
          </section>

          {/* buttons */}
          <section className="flex h-[50px] items-center justify-end gap-3 py-8">
            <Button
              type="button"
              variant="secondary"
              className="h-[50px] rounded-[35px] bg-neutral-200 px-6 py-3 font-sans text-base font-medium text-neutral-800 transition-colors hover:bg-neutral-300 dark:border-neutral-700 dark:bg-neutral-700 dark:text-neutral-200 dark:hover:border-neutral-600 dark:hover:bg-neutral-600"
              onClick={() => {
                setProfileData(profile);
              }}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isSubmitting}
              className="h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 font-sans text-base font-medium text-white transition-colors hover:bg-neutral-900 dark:bg-neutral-200 dark:text-neutral-900 dark:hover:bg-neutral-100"
              onClick={submitForm}
            >
              {isSubmitting ? "Saving..." : "Save changes"}
            </Button>
          </section>
        </form>
      </div>
    </div>
  );
};
