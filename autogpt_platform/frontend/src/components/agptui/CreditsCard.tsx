import { IconRefresh } from '@/components/ui/icons'

interface CreditsCardProps {
  credits: number
  onRefresh?: () => void
}

const CreditsCard = ({ credits, onRefresh }: CreditsCardProps) => {
  return (
    <div className="h-[60px] p-4 bg-neutral-200 rounded-2xl inline-flex items-center gap-2.5">
      <div className="flex items-center gap-0.5">
        <span className="text-neutral-900 text-base font-semibold font-['Geist'] leading-7">
          {credits.toLocaleString()}
        </span>
        <span className="text-neutral-900 text-base font-normal font-['Geist'] leading-7">
          credits
        </span>
      </div>
      <button
        onClick={onRefresh}
          className="w-6 h-6 hover:text-neutral-700 transition-colors"
          aria-label="Refresh credits"
        >
          <IconRefresh className="w-6 h-6" />
      </button>
    </div>
  )
}

export default CreditsCard