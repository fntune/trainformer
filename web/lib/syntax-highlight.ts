import { createHighlighter, type Highlighter } from "shiki"

let highlighter: Highlighter | null = null

export async function getHighlighter(): Promise<Highlighter> {
  if (!highlighter) {
    highlighter = await createHighlighter({
      themes: ["one-dark-pro"],
      langs: ["python", "bash", "typescript", "tsx"],
    })
  }
  return highlighter
}

export async function highlightCode(code: string, lang: string): Promise<string> {
  const hl = await getHighlighter()
  return hl.codeToHtml(code, {
    lang,
    theme: "one-dark-pro",
  })
}
