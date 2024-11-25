"use client";

import * as React from "react";
import { Button } from "./Button";
import Image from "next/image";
import { IconPersonFill } from "@/components/ui/icons";
import { useState } from "react";
import AutoGPTServerAPIServerSide from "@/lib/autogpt-server-api/client";
import { CreatorDetails } from "@/lib/autogpt-server-api/";
export const AVAILABLE_CATEGORIES = [
  "Entertainment",
  "Blog",
  "Business",
  "Content creation",
  "AI Development",
  "Tech",
  "Open Source",
  "Education",
  "Research",
] as const;

export const ProfileInfoForm = ({ profile }: { profile: CreatorDetails }) => {
  return (
    <div className="w-full p-4 sm:p-8">
      <h1 className="mb-6 font-['Poppins'] text-[28px] font-medium text-neutral-900 dark:text-neutral-100 sm:mb-8 sm:text-[35px]">
        Profile
      </h1>

      <div className="mb-8 sm:mb-12">
        <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row sm:items-start">
          <div className="relative h-[130px] w-[130px] rounded-full bg-[#d9d9d9] dark:bg-[#333333]">
            {profile.avatar_url ? (
              <Image
                src={profile.avatar_url}
                alt="Profile"
                fill
                className="rounded-full"
              />
            ) : (
              <IconPersonFill className="absolute left-[30px] top-[24px] h-[77.80px] w-[70.63px] text-[#7e7e7e] dark:text-[#999999]" />
            )}
          </div>
          <Button
            variant="default"
            className="mt-11 h-[43px] rounded-[22px] border border-slate-900 bg-slate-900 px-4 py-2 font-['Inter'] text-sm font-medium text-slate-50 transition-colors hover:bg-white hover:text-slate-900 dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100"
            onClick={() => {
              const input = document.createElement("input");
              input.type = "file";
              input.accept = "image/*";
              input.onchange = async (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (file) {
                  try {
                    const api = new AutoGPTServerAPIServerSide();
                    await api.uploadStoreSubmissionMedia(file);
                  } catch (error) {
                    console.error("Error uploading image:", error);
                  }
                }
              };
              input.click();
            }}
          >
            Edit photo
          </Button>
        </div>

        <form
          className="space-y-4 sm:space-y-6"
          onSubmit={async (e) => {
            e.preventDefault();
            const formData = new FormData(e.currentTarget);

            try {
              const api = new AutoGPTServerAPIServerSide();
              await api.updateStoreProfile({
                name: formData.get("displayName") as string,
                username: formData.get("handle") as string,
                description: formData.get("bio") as string,
                avatar_url: profile.avatar_url || "",
                links: [1, 2, 3, 4, 5]
                  .map((num) => formData.get(`link${num}`) as string)
                  .filter(Boolean),
                agent_rating: 0,
                agent_runs: 0,
                top_categories: [],
              });
            } catch (error) {
              console.error("Error updating profile:", error);
            }
          }}
        >
          <div className="w-full">
            <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
              Display name
            </label>
            <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700">
              <input
                type="text"
                name="displayName"
                defaultValue={profile.name}
                placeholder="Enter your display name"
                className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
              />
            </div>
          </div>

          <div className="w-full">
            <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
              Handle
            </label>
            <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700">
              <input
                type="text"
                name="handle"
                defaultValue={profile.username}
                placeholder="@username"
                className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
              />
            </div>
          </div>

          <div className="w-full">
            <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
              Bio
            </label>
            <div className="h-[220px] rounded-2xl border border-slate-200 py-2.5 pl-4 pr-4 dark:border-slate-700">
              <textarea
                name="bio"
                defaultValue={profile.description}
                placeholder="Tell us about yourself..."
                className="h-full w-full resize-none border-none bg-transparent font-['Geist'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
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
                const link = profile.links[linkNum - 1];
                return (
                  <div key={linkNum} className="w-full">
                    <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950 dark:text-slate-50">
                      Link {linkNum}
                    </label>
                    <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700">
                      <input
                        type="text"
                        name={`link${linkNum}`}
                        placeholder="https://"
                        defaultValue={link || ""}
                        className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none dark:text-[#999999]"
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <hr className="my-8 border-neutral-300 dark:border-neutral-700" />

          <div className="flex h-[50px] items-center justify-end gap-3">
            <Button
              type="button"
              variant="secondary"
              className="h-[50px] rounded-[35px] bg-neutral-200 px-6 py-3 font-['Geist'] text-base font-medium text-neutral-800 transition-colors hover:bg-neutral-300 dark:border-neutral-700 dark:bg-neutral-700 dark:text-neutral-200 dark:hover:border-neutral-600 dark:hover:bg-neutral-600"
              onClick={(e) => {
                e.preventDefault();
                const form = e.currentTarget.closest("form");
                if (form) form.reset();
              }}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="default"
              className="h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 font-['Geist'] text-base font-medium text-white transition-colors hover:bg-neutral-900 dark:bg-neutral-200 dark:text-neutral-900 dark:hover:bg-neutral-100"
              onClick={(e) => {
                e.preventDefault();
                const form = e.currentTarget.closest("form");
                if (form) form.submit();
              }}
            >
              Save changes
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
