import * as React from "react";
import { Button } from "./Button";
import Image from "next/image";
import { IconPersonFill } from "@/components/ui/icons";

export interface ProfileInfoFormProps {
  displayName: string;
  handle: string;
  bio: string;
  profileImage?: string;
  links: { id: number; url: string }[];
  categories: { id: number; name: string }[];
}

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

export const ProfileInfoForm: React.FC<ProfileInfoFormProps> = ({
  displayName,
  handle,
  bio,
  profileImage,
  links,
  categories,
}) => {
  const [selectedCategories, setSelectedCategories] = React.useState<string[]>(
    categories.map((cat) => cat.name),
  );

  const handleCategoryClick = (category: string) => {
    console.log(`${category} category button was pressed`);
    if (selectedCategories.includes(category)) {
      setSelectedCategories(selectedCategories.filter((c) => c !== category));
    } else if (selectedCategories.length < 5) {
      setSelectedCategories([...selectedCategories, category]);
    }
  };

  return (
    <main className="p-4 sm:p-8">
      <h1 className="mb-6 font-['Poppins'] text-[28px] font-medium text-neutral-900 sm:mb-8 sm:text-[35px]">
        Profile
      </h1>

      <div className="mb-8 sm:mb-12">
        <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row sm:items-start">
          <div className="relative h-[130px] w-[130px] rounded-full bg-[#d9d9d9]">
            {profileImage ? (
              <Image
                src={profileImage}
                alt="Profile"
                layout="fill"
                className="rounded-full"
              />
            ) : (
              <IconPersonFill className="absolute left-[30px] top-[24px] h-[77.80px] w-[70.63px] text-[#7e7e7e]" />
            )}
          </div>
          <Button
            variant="default"
            className="mt-11 h-[43px] rounded-[22px] border border-slate-900 bg-slate-900 px-4 py-2 font-['Inter'] text-sm font-medium text-slate-50 transition-colors hover:bg-white hover:text-slate-900"
          >
            Edit photo
          </Button>
        </div>

        <div className="space-y-4 sm:space-y-6">
          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:gap-0">
            <div className="w-full sm:w-[638px]">
              <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950">
                Display name
              </label>
              <div className="rounded-[55px] border border-slate-200 px-4 py-2.5">
                <input
                  type="text"
                  value={displayName}
                  placeholder="Enter your display name"
                  className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none"
                />
              </div>
            </div>
            <Button
              variant="default"
              className="h-[43px] rounded-[22px] border border-slate-900 bg-slate-900 px-4 py-2 font-['Inter'] text-sm font-medium text-slate-50 transition-colors hover:bg-white hover:text-slate-900"
            >
              Edit
            </Button>
          </div>

          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:gap-0">
            <div className="w-full sm:w-[638px]">
              <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950">
                Handle
              </label>
              <div className="rounded-[55px] border border-slate-200 px-4 py-2.5">
                <input
                  type="text"
                  value={handle}
                  placeholder="@username"
                  className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none"
                />
              </div>
            </div>
            <Button
              variant="default"
              className="h-[43px] rounded-[22px] border border-slate-900 bg-slate-900 px-4 py-2 font-['Inter'] text-sm font-medium text-slate-50 transition-colors hover:bg-white hover:text-slate-900"
            >
              Edit
            </Button>
          </div>

          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:gap-0">
            <div className="w-full sm:w-[638px]">
              <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950">
                Bio
              </label>
              <div className="h-[220px] rounded-2xl border border-slate-200 py-2.5 pl-4 pr-4">
                <textarea
                  value={bio}
                  placeholder="Tell us about yourself..."
                  className="h-full w-full resize-none border-none bg-transparent font-['Geist'] text-base font-normal text-[#666666] focus:outline-none"
                />
              </div>
            </div>
            <Button
              variant="default"
              className="h-[43px] rounded-[22px] border border-slate-900 bg-slate-900 px-4 py-2 font-['Inter'] text-sm font-medium text-slate-50 transition-colors hover:bg-white hover:text-slate-900"
            >
              Edit
            </Button>
          </div>
        </div>
      </div>

      <hr className="my-8 border-neutral-300" />

      <section className="mb-8">
        <h2 className="mb-4 font-['Poppins'] text-lg font-semibold leading-7 text-neutral-500">
          Your links
        </h2>
        <p className="mb-6 font-['Geist'] text-base font-medium leading-tight text-slate-950">
          You can display up to 5 links on your profile
        </p>

        <div className="space-y-4 sm:space-y-6">
          {[1, 2, 3, 4, 5].map((linkNum) => {
            const link = links.find((l) => l.id === linkNum);
            return (
              <div
                key={linkNum}
                className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:gap-0"
              >
                <div className="w-full sm:w-[638px]">
                  <label className="mb-1.5 block font-['Geist'] text-base font-medium leading-tight text-slate-950">
                    Link {linkNum}
                  </label>
                  <div className="rounded-[55px] border border-slate-200 px-4 py-2.5">
                    <input
                      type="text"
                      placeholder="https://"
                      value={link?.url || ""}
                      className="w-full border-none bg-transparent font-['Inter'] text-base font-normal text-[#666666] focus:outline-none"
                    />
                  </div>
                </div>
                <Button
                  variant="default"
                  className={`h-[43px] px-4 py-2 ${
                    link?.url
                      ? "border border-slate-900 bg-slate-900 text-slate-50 hover:bg-white hover:text-slate-900"
                      : "bg-gray-500 text-slate-50 hover:bg-gray-600"
                  } rounded-[22px] font-['Inter'] text-sm font-medium transition-colors`}
                >
                  Save
                </Button>
              </div>
            );
          })}
        </div>
      </section>

      <hr className="my-8 border-neutral-300" />

      <section>
        <div className="relative min-h-[190px] w-full">
          <div className="absolute left-0 top-0 h-[68px] w-full">
            <div className="absolute left-0 top-[48px] w-full font-['Geist'] text-base font-medium leading-tight text-slate-950">
              Pick up to 5 categories for your profile
            </div>
            <div className="absolute left-0 top-0 font-['Poppins'] text-lg font-semibold leading-7 text-neutral-500">
              Categories
            </div>
          </div>
          <div className="absolute left-0 top-[84px] inline-flex w-full flex-wrap items-center justify-start gap-2 sm:gap-2.5">
            {AVAILABLE_CATEGORIES.map((category, index) => (
              <Button
                key={index}
                variant="outline"
                onClick={() => handleCategoryClick(category)}
                className={`rounded-[34px] border border-neutral-600 px-5 py-3 transition-colors ${
                  selectedCategories.includes(category)
                    ? "bg-slate-900 text-white hover:bg-slate-800"
                    : "text-neutral-800 hover:border-neutral-800 hover:bg-neutral-100"
                } font-['Geist'] text-base font-normal leading-normal`}
              >
                {category}
              </Button>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
};
