"use client"

import { useEffect, useState } from "react"
import { highlightCode } from "@/lib/syntax-highlight"

interface HighlightedCodeProps {
  code: string
  lang: string
}

export default function HighlightedCode({ code, lang }: HighlightedCodeProps) {
  const [html, setHtml] = useState<string>("")

  useEffect(() => {
    highlightCode(code, lang).then(setHtml)
  }, [code, lang])

  if (!html) {
    return (
      <pre className="p-4 bg-code-bg rounded-lg overflow-x-auto animate-pulse">
        <code className="text-sm font-mono text-muted">{code}</code>
      </pre>
    )
  }

  return (
    <div
      className="[&_pre]:p-4 [&_pre]:bg-code-bg [&_pre]:rounded-lg [&_pre]:overflow-x-auto [&_code]:text-sm [&_code]:font-mono"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
