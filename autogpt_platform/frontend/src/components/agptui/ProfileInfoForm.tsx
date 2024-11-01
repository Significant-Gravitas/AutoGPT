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
  "Research"
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
    categories.map(cat => cat.name)
  );

  const handleCategoryClick = (category: string) => {
    console.log(`${category} category button was pressed`);
    if (selectedCategories.includes(category)) {
      setSelectedCategories(selectedCategories.filter(c => c !== category));
    } else if (selectedCategories.length < 5) {
      setSelectedCategories([...selectedCategories, category]);
    }
  };

  return (
    <main className="p-4 sm:p-8">
      <h1 className="text-[28px] sm:text-[35px] font-medium font-['Poppins'] text-neutral-900 mb-6 sm:mb-8">
        Profile
      </h1>

      <div className="mb-8 sm:mb-12">
        <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4 mb-8">
          <div className="w-[130px] h-[130px] relative bg-[#d9d9d9] rounded-full">
            {profileImage ? (
              <Image
                src={profileImage}
                alt="Profile"
                layout="fill"
                className="rounded-full"
              />
            ) : (
              <IconPersonFill className="w-[70.63px] h-[77.80px] absolute left-[30px] top-[24px] text-[#7e7e7e]" />
            )}
          </div>
          <Button 
            variant="default" 
            className="h-[43px] px-4 py-2 mt-11 bg-slate-900 hover:bg-white hover:text-slate-900 rounded-[22px] text-slate-50 text-sm font-medium font-['Inter'] border border-slate-900 transition-colors"
          >
            Edit photo
          </Button>
        </div>

        <div className="space-y-4 sm:space-y-6">
          <div className="flex flex-col sm:flex-row justify-between items-start gap-4 sm:gap-0">
            <div className="w-full sm:w-[638px]">
              <label className="text-slate-950 text-base font-medium font-['Geist'] leading-tight mb-1.5 block">
                Display name
              </label>
              <div className="px-4 py-2.5 rounded-[55px] border border-slate-200">
                <input
                  type="text"
                  value={displayName}
                  placeholder="Enter your display name"
                  className="w-full text-[#666666] text-base font-normal font-['Inter'] bg-transparent border-none focus:outline-none"
                />
              </div>
            </div>
            <Button 
              variant="default" 
              className="h-[43px] px-4 py-2 bg-slate-900 hover:bg-white hover:text-slate-900 rounded-[22px] text-slate-50 text-sm font-medium font-['Inter'] border border-slate-900 transition-colors"
            >
              Edit
            </Button>
          </div>

          <div className="flex flex-col sm:flex-row justify-between items-start gap-4 sm:gap-0">
            <div className="w-full sm:w-[638px]">
              <label className="text-slate-950 text-base font-medium font-['Geist'] leading-tight mb-1.5 block">
                Handle
              </label>
              <div className="px-4 py-2.5 rounded-[55px] border border-slate-200">
                <input
                  type="text"
                  value={handle}
                  placeholder="@username"
                  className="w-full text-[#666666] text-base font-normal font-['Inter'] bg-transparent border-none focus:outline-none"
                />
              </div>
            </div>
            <Button 
              variant="default" 
              className="h-[43px] px-4 py-2 bg-slate-900 hover:bg-white hover:text-slate-900 rounded-[22px] text-slate-50 text-sm font-medium font-['Inter'] border border-slate-900 transition-colors"
            >
              Edit
            </Button>
          </div>

          <div className="flex flex-col sm:flex-row justify-between items-start gap-4 sm:gap-0">
            <div className="w-full sm:w-[638px]">
              <label className="text-slate-950 text-base font-medium font-['Geist'] leading-tight mb-1.5 block">
                Bio
              </label>
              <div className="pl-4 pr-4 py-2.5 rounded-2xl border border-slate-200 h-[220px]">
                <textarea
                  value={bio}
                  placeholder="Tell us about yourself..."
                  className="w-full h-full text-[#666666] text-base font-normal font-['Geist'] bg-transparent border-none focus:outline-none resize-none"
                />
              </div>
            </div>
            <Button 
              variant="default" 
              className="h-[43px] px-4 py-2 bg-slate-900 hover:bg-white hover:text-slate-900 rounded-[22px] text-slate-50 text-sm font-medium font-['Inter'] border border-slate-900 transition-colors"
            >
              Edit
            </Button>
          </div>
        </div>
      </div>

      <hr className="border-neutral-300 my-8" />

      <section className="mb-8">
        <h2 className="text-neutral-500 text-lg font-semibold font-['Poppins'] leading-7 mb-4">
          Your links
        </h2>
        <p className="text-slate-950 text-base font-medium font-['Geist'] leading-tight mb-6">
          You can display up to 5 links on your profile
        </p>

        <div className="space-y-4 sm:space-y-6">
          {[1, 2, 3, 4, 5].map((linkNum) => {
            const link = links.find(l => l.id === linkNum);
            return (
              <div key={linkNum} className="flex flex-col sm:flex-row justify-between items-start gap-4 sm:gap-0">
                <div className="w-full sm:w-[638px]">
                  <label className="text-slate-950 text-base font-medium font-['Geist'] leading-tight mb-1.5 block">
                    Link {linkNum}
                  </label>
                  <div className="px-4 py-2.5 rounded-[55px] border border-slate-200">
                    <input
                      type="text"
                      placeholder="https://"
                      value={link?.url || ""}
                      className="w-full text-[#666666] text-base font-normal font-['Inter'] bg-transparent border-none focus:outline-none"
                    />
                  </div>
                </div>
                <Button 
                  variant="default" 
                  className={`h-[43px] px-4 py-2 
                    ${link?.url 
                      ? 'bg-slate-900 text-slate-50 hover:bg-white hover:text-slate-900 border border-slate-900' 
                      : 'bg-gray-500 hover:bg-gray-600 text-slate-50'} 
                    rounded-[22px] text-sm font-medium font-['Inter'] transition-colors`}
                >
                  Save
                </Button>
              </div>
            );
          })}
        </div>
      </section>

      <hr className="border-neutral-300 my-8" />

      <section>
        <div className="w-full min-h-[190px] relative">
          <div className="w-full h-[68px] left-0 top-0 absolute">
            <div className="w-full left-0 top-[48px] absolute text-slate-950 text-base font-medium font-['Geist'] leading-tight">
              Pick up to 5 categories for your profile
            </div>
            <div className="left-0 top-0 absolute text-neutral-500 text-lg font-semibold font-['Poppins'] leading-7">
              Categories
            </div>
          </div>
          <div className="w-full left-0 top-[84px] absolute justify-start items-center gap-2 sm:gap-2.5 inline-flex flex-wrap">
            {AVAILABLE_CATEGORIES.map((category, index) => (
              <Button
                key={index}
                variant="outline"
                onClick={() => handleCategoryClick(category)}
                className={`px-5 py-3 rounded-[34px] border border-neutral-600 transition-colors
                  ${selectedCategories.includes(category) 
                    ? 'bg-slate-900 text-white hover:bg-slate-800' 
                    : 'text-neutral-800 hover:bg-neutral-100 hover:border-neutral-800'} 
                  text-base font-normal font-['Geist'] leading-normal`}
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
