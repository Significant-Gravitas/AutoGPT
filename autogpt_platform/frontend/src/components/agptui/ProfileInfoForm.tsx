"use client";

import * as React from "react";
import { useState, useRef } from "react";

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
import AutogptButton from "./AutogptButton";
import AutogptInput from "./AutogptInput";

export const ProfileInfoForm = ({ profile }: { profile: CreatorDetails }) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [profileData, setProfileData] = useState(profile);
  const { supabase } = useSupabase();
  const api = useBackendAPI();
  const editPhotoRef = useRef<HTMLInputElement>(null);

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
    <div className="mb-8 sm:mb-12">
      <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row">
        <div className="flex h-[6.25rem] w-[6.25rem] items-center justify-center rounded-full bg-[#DADADA]">
          {profileData.avatar_url ? (
            <Image
              src={profileData.avatar_url}
              alt="Profile"
              fill
              className="rounded-full"
            />
          ) : (
            <IconPersonFill className="h-10 w-10 text-[#7e7e7e]" />
          )}
        </div>
        <div>
          <Input
            type="file"
            accept="image/*"
            className="hidden"
            ref={editPhotoRef}
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (file) {
                await handleImageUpload(file);
              }
            }}
          />
          <AutogptButton onClick={() => editPhotoRef.current?.click()}>
            Edit photo
          </AutogptButton>
        </div>
      </div>

      <form className="space-y-10" onSubmit={submitForm}>
        {/* Top section */}
        <section className="max-w-3xl space-y-6">
          <AutogptInput
            label="Display name"
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

          <AutogptInput
            label="Handle"
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

          <div className="w-full space-y-1.5">
            <Label className="font-sans text-sm font-medium leading-[1.4rem]">
              Bio
            </Label>
            <Textarea
              name="bio"
              defaultValue={profileData.description}
              placeholder="Tell us about yourself..."
              className="m-0 h-10 min-h-56 w-full resize-none rounded-3xl border border-zinc-300 bg-white py-2 pl-4 font-sans text-base font-normal text-zinc-800 shadow-none outline-none placeholder:text-zinc-400 focus:border-2 focus:border-[#CBD5E1] focus:shadow-none focus:ring-0"
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
        <section className="mb-8 max-w-3xl space-y-6">
          <div>
            <h2 className="font-poppins text-base font-medium text-neutral-900">
              Your links
            </h2>
            <p className="font-sans text-sm font-normal text-zinc-800">
              You can display up to 5 links on your profile
            </p>
          </div>

          <div className="space-y-4 sm:space-y-6">
            {[1, 2, 3, 4, 5].map((linkNum) => {
              const link = profileData.links[linkNum - 1];
              return (
                <AutogptInput
                  key={linkNum}
                  label={`Link ${linkNum}`}
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
              );
            })}
          </div>
        </section>

        {/* buttons */}
        <section className="flex h-[50px] items-center justify-end gap-3 py-8">
          <AutogptButton
            type="button"
            variant="secondary"
            className="h-[50px] rounded-[35px] bg-neutral-200 px-6 py-3 font-sans text-base font-medium text-neutral-800 transition-colors hover:bg-neutral-300 dark:border-neutral-700 dark:bg-neutral-700 dark:text-neutral-200 dark:hover:border-neutral-600 dark:hover:bg-neutral-600"
            onClick={() => {
              setProfileData(profile);
            }}
          >
            Cancel
          </AutogptButton>
          <AutogptButton
            type="submit"
            disabled={isSubmitting}
            className="h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 font-sans text-base font-medium text-white transition-colors hover:bg-neutral-900 dark:bg-neutral-200 dark:text-neutral-900 dark:hover:bg-neutral-100"
            onClick={submitForm}
          >
            {isSubmitting ? "Saving..." : "Save changes"}
          </AutogptButton>
        </section>
      </form>
    </div>
  );
};
