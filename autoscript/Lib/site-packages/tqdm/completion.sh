#!/usr/bin/env bash
_tqdm(){
  local cur prv
  cur="${COMP_WORDS[COMP_CWORD]}"
  prv="${COMP_WORDS[COMP_CWORD - 1]}"

  case ${prv} in
  --bar_format|--buf_size|--colour|--comppath|--delay|--delim|--desc|--initial|--lock_args|--manpath|--maxinterval|--mininterval|--miniters|--ncols|--nrows|--position|--postfix|--smoothing|--total|--unit|--unit_divisor)
    # await user input
    ;;
  "--log")
    COMPREPLY=($(compgen -W       'CRITICAL FATAL ERROR WARN WARNING INFO DEBUG NOTSET' -- ${cur}))
    ;;
  *)
    COMPREPLY=($(compgen -W '--ascii --bar_format --buf_size --bytes --colour --comppath --delay --delim --desc --disable --dynamic_ncols --help --initial --leave --lock_args --log --manpath --maxinterval --mininterval --miniters --ncols --nrows --null --position --postfix --smoothing --tee --total --unit --unit_divisor --unit_scale --update --update_to --version --write_bytes -h -v' -- ${cur}))
    ;;
  esac
}
complete -F _tqdm tqdm
