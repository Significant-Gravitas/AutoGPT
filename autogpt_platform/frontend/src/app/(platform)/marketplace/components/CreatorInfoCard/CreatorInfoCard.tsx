import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Text } from "@/components/atoms/Text/Text";
import { StarRating } from "../StarRating/StarRating";

interface Props {
  username: string;
  handle: string;
  avatarSrc: string | null;
  categories: string[];
  averageRating: number;
  totalRuns: number;
}

export function CreatorInfoCard({
  username,
  handle,
  avatarSrc,
  categories,
  averageRating,
  totalRuns,
}: Props) {
  return (
    <div
      className="inline-flex h-auto min-h-[500px] w-full max-w-[440px] flex-col items-start justify-between rounded-[26px] bg-purple-100 p-4 sm:h-[632px] sm:w-[440px] sm:p-6"
      role="article"
      aria-label={`Creator profile for ${username}`}
    >
      <div className="flex w-full flex-col items-start justify-start gap-3.5 sm:h-[218px]">
        <Avatar className="h-[100px] w-[100px] sm:h-[130px] sm:w-[130px]">
          {avatarSrc && (
            <AvatarImage
              width={130}
              height={130}
              src={avatarSrc}
              alt={`${username}'s avatar`}
            />
          )}
          <AvatarFallback
            size={130}
            className="h-[100px] w-[100px] sm:h-[130px] sm:w-[130px]"
          >
            {username}
          </AvatarFallback>
        </Avatar>
        <div className="flex w-full flex-col items-start justify-start gap-1.5">
          <Text
            variant="h2"
            as="div"
            tone="primary"
            unmask={false}
            data-testid="creator-title"
            className="w-full text-[35px] leading-10 tracking-normal"
          >
            {username}
          </Text>
          <Text
            variant="lead"
            as="div"
            unmask={false}
            className="w-full text-lg leading-6 text-zinc-800 sm:text-xl sm:leading-7"
          >
            @{handle}
          </Text>
        </div>
      </div>

      <div className="my-4 flex w-full flex-col items-start justify-start gap-6 sm:gap-[50px]">
        <div className="flex w-full flex-col items-start justify-start gap-3">
          <div className="h-px w-full bg-zinc-700" />
          <div className="flex flex-col items-start justify-start gap-2.5">
            <Text
              variant="large-medium"
              as="div"
              className="w-full leading-normal text-zinc-800"
            >
              Top categories
            </Text>
            <div
              className="flex flex-wrap items-center gap-2.5"
              role="list"
              aria-label="Categories"
            >
              {categories.map((category, index) => (
                <div
                  key={index}
                  className="flex items-center justify-center gap-2.5 rounded-full border border-zinc-600 px-4 py-3"
                  role="listitem"
                >
                  <Text
                    variant="large"
                    as="div"
                    unmask={false}
                    className="leading-normal text-zinc-800"
                  >
                    {category}
                  </Text>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex w-full flex-col items-start justify-start gap-3">
          <div className="h-px w-full bg-zinc-700" />
          <div className="flex w-full flex-col items-start justify-between gap-4 sm:flex-row sm:gap-0">
            <div className="flex w-full flex-col items-start justify-start gap-2.5 sm:w-[164px]">
              <Text
                variant="large-medium"
                as="div"
                className="w-full leading-normal text-zinc-800"
              >
                Average rating
              </Text>
              <div className="inline-flex items-center gap-2">
                <Text
                  variant="large-semibold"
                  as="div"
                  className="text-lg leading-7 text-zinc-800"
                >
                  {averageRating.toFixed(1)}
                </Text>
                <div
                  className="flex items-center gap-px"
                  role="img"
                  aria-label={`Rating: ${averageRating} out of 5 stars`}
                >
                  <StarRating rating={averageRating} />
                </div>
              </div>
            </div>
            <div className="flex w-full flex-col items-start justify-start gap-2.5 sm:w-[164px]">
              <Text
                variant="large-medium"
                as="div"
                className="w-full leading-normal text-zinc-800"
              >
                Number of runs
              </Text>
              <Text
                variant="large-semibold"
                as="div"
                className="text-lg leading-7 text-zinc-800"
              >
                {new Intl.NumberFormat().format(totalRuns)} runs
              </Text>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
