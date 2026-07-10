import { FolderIcon, FolderOpenIcon } from "@/components/atoms/AGPTIcon/icons";
import { cn } from "@/lib/utils";
import astro from "material-icon-theme/icons/astro.svg";
import audio from "material-icon-theme/icons/audio.svg";
import c from "material-icon-theme/icons/c.svg";
import clojure from "material-icon-theme/icons/clojure.svg";
import shell from "material-icon-theme/icons/console.svg";
import cpp from "material-icon-theme/icons/cpp.svg";
import csharp from "material-icon-theme/icons/csharp.svg";
import css from "material-icon-theme/icons/css.svg";
import dart from "material-icon-theme/icons/dart.svg";
import database from "material-icon-theme/icons/database.svg";
import docker from "material-icon-theme/icons/docker.svg";
import plainText from "material-icon-theme/icons/document.svg";
import elixir from "material-icon-theme/icons/elixir.svg";
import elm from "material-icon-theme/icons/elm.svg";
import eslint from "material-icon-theme/icons/eslint.svg";
import file from "material-icon-theme/icons/file.svg";
import font from "material-icon-theme/icons/font.svg";
import git from "material-icon-theme/icons/git.svg";
import go from "material-icon-theme/icons/go.svg";
import goMod from "material-icon-theme/icons/go-mod.svg";
import graphql from "material-icon-theme/icons/graphql.svg";
import h from "material-icon-theme/icons/h.svg";
import hpp from "material-icon-theme/icons/hpp.svg";
import html from "material-icon-theme/icons/html.svg";
import image from "material-icon-theme/icons/image.svg";
import java from "material-icon-theme/icons/java.svg";
import javascript from "material-icon-theme/icons/javascript.svg";
import json from "material-icon-theme/icons/json.svg";
import kotlin from "material-icon-theme/icons/kotlin.svg";
import less from "material-icon-theme/icons/less.svg";
import license from "material-icon-theme/icons/license.svg";
import lock from "material-icon-theme/icons/lock.svg";
import log from "material-icon-theme/icons/log.svg";
import lua from "material-icon-theme/icons/lua.svg";
import makefile from "material-icon-theme/icons/makefile.svg";
import markdown from "material-icon-theme/icons/markdown.svg";
import mdx from "material-icon-theme/icons/mdx.svg";
import next from "material-icon-theme/icons/next.svg";
import nodejs from "material-icon-theme/icons/nodejs.svg";
import pdf from "material-icon-theme/icons/pdf.svg";
import php from "material-icon-theme/icons/php.svg";
import pnpm from "material-icon-theme/icons/pnpm.svg";
import powershell from "material-icon-theme/icons/powershell.svg";
import prettier from "material-icon-theme/icons/prettier.svg";
import prisma from "material-icon-theme/icons/prisma.svg";
import proto from "material-icon-theme/icons/proto.svg";
import python from "material-icon-theme/icons/python.svg";
import pythonMisc from "material-icon-theme/icons/python-misc.svg";
import r from "material-icon-theme/icons/r.svg";
import react from "material-icon-theme/icons/react.svg";
import reactTs from "material-icon-theme/icons/react_ts.svg";
import readme from "material-icon-theme/icons/readme.svg";
import ruby from "material-icon-theme/icons/ruby.svg";
import rust from "material-icon-theme/icons/rust.svg";
import sass from "material-icon-theme/icons/sass.svg";
import scala from "material-icon-theme/icons/scala.svg";
import settings from "material-icon-theme/icons/settings.svg";
import svg from "material-icon-theme/icons/svg.svg";
import svelte from "material-icon-theme/icons/svelte.svg";
import swift from "material-icon-theme/icons/swift.svg";
import table from "material-icon-theme/icons/table.svg";
import toml from "material-icon-theme/icons/toml.svg";
import tsconfig from "material-icon-theme/icons/tsconfig.svg";
import tune from "material-icon-theme/icons/tune.svg";
import typescript from "material-icon-theme/icons/typescript.svg";
import video from "material-icon-theme/icons/video.svg";
import vue from "material-icon-theme/icons/vue.svg";
import xml from "material-icon-theme/icons/xml.svg";
import yaml from "material-icon-theme/icons/yaml.svg";
import yarn from "material-icon-theme/icons/yarn.svg";
import zip from "material-icon-theme/icons/zip.svg";

// Static SVG imports resolve to `{ src }` under webpack but to a plain URL
// string under Turbopack — support both so the icons render in dev and prod.
type IconAsset = { src: string } | string;

function iconSrc(asset: IconAsset): string {
  return typeof asset === "string" ? asset : asset.src;
}

// Genuine VS Code file icons (Material Icon Theme). Matched full-filename first,
// then by extension, falling back to the default document glyph.
const BY_NAME: Record<string, IconAsset> = {
  dockerfile: docker,
  "package.json": nodejs,
  "package-lock.json": nodejs,
  "pnpm-lock.yaml": pnpm,
  "yarn.lock": yarn,
  "tsconfig.json": tsconfig,
  "next.config.js": next,
  "next.config.mjs": next,
  "next.config.ts": next,
  ".gitignore": git,
  ".gitattributes": git,
  ".eslintrc": eslint,
  ".prettierrc": prettier,
  "readme.md": readme,
  license: license,
  "license.md": license,
  makefile: makefile,
  "requirements.txt": pythonMisc,
  "pyproject.toml": pythonMisc,
  "cargo.toml": rust,
  "go.mod": goMod,
};

const BY_EXTENSION: Record<string, IconAsset> = {
  js: javascript,
  mjs: javascript,
  cjs: javascript,
  jsx: react,
  ts: typescript,
  mts: typescript,
  cts: typescript,
  tsx: reactTs,
  py: python,
  rb: ruby,
  go: go,
  rs: rust,
  java: java,
  kt: kotlin,
  kts: kotlin,
  c: c,
  h: h,
  cpp: cpp,
  cc: cpp,
  hpp: hpp,
  cs: csharp,
  php: php,
  swift: swift,
  dart: dart,
  lua: lua,
  r: r,
  scala: scala,
  clj: clojure,
  ex: elixir,
  exs: elixir,
  elm: elm,
  json: json,
  jsonc: json,
  json5: json,
  yaml: yaml,
  yml: yaml,
  toml: toml,
  xml: xml,
  html: html,
  htm: html,
  css: css,
  scss: sass,
  sass: sass,
  less: less,
  md: markdown,
  markdown: markdown,
  mdx: mdx,
  txt: plainText,
  log: log,
  csv: table,
  tsv: table,
  sh: shell,
  bash: shell,
  zsh: shell,
  fish: shell,
  bat: shell,
  ps1: powershell,
  sql: database,
  graphql: graphql,
  gql: graphql,
  prisma: prisma,
  proto: proto,
  vue: vue,
  svelte: svelte,
  astro: astro,
  svg: svg,
  png: image,
  jpg: image,
  jpeg: image,
  gif: image,
  webp: image,
  ico: image,
  bmp: image,
  pdf: pdf,
  zip: zip,
  tar: zip,
  gz: zip,
  rar: zip,
  "7z": zip,
  mp4: video,
  webm: video,
  mp3: audio,
  wav: audio,
  ttf: font,
  otf: font,
  woff: font,
  woff2: font,
  env: tune,
  ini: settings,
  conf: settings,
  lock: lock,
};

function resolveIcon(name: string): IconAsset {
  const lower = name.toLowerCase();
  if (BY_NAME[lower]) return BY_NAME[lower];
  const ext = lower.includes(".") ? (lower.split(".").pop() ?? "") : "";
  return BY_EXTENSION[ext] ?? file;
}

interface Props {
  name: string;
  size?: number;
  className?: string;
  isDir?: boolean;
  isOpen?: boolean;
}

export function FileTypeIcon({
  name,
  size = 14,
  className,
  isDir = false,
  isOpen = false,
}: Props) {
  // Folders use the flag-aware (Pikaicons) line folder; files keep the
  // Material Icon Theme glyphs so language/type is recognisable at a glance.
  if (isDir) {
    const Folder = isOpen ? FolderOpenIcon : FolderIcon;
    return (
      <Folder size={size} className={cn("shrink-0 text-zinc-500", className)} />
    );
  }

  return (
    <img
      src={iconSrc(resolveIcon(name))}
      alt=""
      width={size}
      height={size}
      className={cn("shrink-0", className)}
      draggable={false}
    />
  );
}
