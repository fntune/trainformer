"use client"

import dynamic from "next/dynamic"
import { CopyButton } from "./CopyButton"

const HighlightedCode = dynamic(() => import("./HighlightedCode"), {
  ssr: false,
  loading: () => (
    <pre className="p-4 bg-code-bg rounded-lg overflow-x-auto animate-pulse">
      <code className="text-sm font-mono text-muted">Loading...</code>
    </pre>
  ),
})

interface CodeBlockProps {
  code: string
  lang: string
  showCopy?: boolean
  title?: string
}

export function CodeBlock({ code, lang, showCopy = true, title }: CodeBlockProps) {
  return (
    <div className="group relative rounded-lg border border-border bg-code-bg overflow-hidden">
      {title ? (
        <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-surface">
          <span className="text-sm font-mono text-muted">{title}</span>
          {showCopy && <CopyButton text={code} />}
        </div>
      ) : null}
      <div className="relative">
        <HighlightedCode code={code} lang={lang} />
        {showCopy && !title ? (
          <div className="absolute top-2 right-2">
            <CopyButton text={code} />
          </div>
        ) : null}
      </div>
    </div>
  )
}
