"use client"

import { type ReactNode, useCallback, useRef, type KeyboardEvent } from "react"
import { useSearchParams, useRouter } from "next/navigation"

interface Tab {
  id: string
  label: string
  content: ReactNode
}

interface TabsProps {
  tabs: Tab[]
  defaultTab?: string
  ariaLabel: string
}

export function Tabs({ tabs, defaultTab, ariaLabel }: TabsProps) {
  const searchParams = useSearchParams()
  const router = useRouter()
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([])

  const activeTab = searchParams.get("tab") ?? defaultTab ?? tabs[0]?.id

  const setTab = useCallback(
    (tabId: string) => {
      const params = new URLSearchParams(searchParams.toString())
      params.set("tab", tabId)
      router.replace(`?${params.toString()}`, { scroll: false })
    },
    [router, searchParams]
  )

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      const currentIndex = tabs.findIndex((t) => t.id === activeTab)
      let nextIndex = currentIndex

      if (e.key === "ArrowRight") {
        nextIndex = (currentIndex + 1) % tabs.length
      } else if (e.key === "ArrowLeft") {
        nextIndex = (currentIndex - 1 + tabs.length) % tabs.length
      } else if (e.key === "Home") {
        nextIndex = 0
      } else if (e.key === "End") {
        nextIndex = tabs.length - 1
      } else {
        return
      }

      e.preventDefault()
      const nextTab = tabs[nextIndex]
      if (nextTab) {
        setTab(nextTab.id)
        tabRefs.current[nextIndex]?.focus()
      }
    },
    [activeTab, tabs, setTab]
  )

  const activeContent = tabs.find((t) => t.id === activeTab)?.content

  return (
    <div>
      <div
        role="tablist"
        aria-label={ariaLabel}
        onKeyDown={handleKeyDown}
        className="flex gap-1 p-1 rounded-lg bg-surface border border-border mb-4"
      >
        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            ref={(el) => {
              tabRefs.current[index] = el
            }}
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => setTab(tab.id)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              activeTab === tab.id
                ? "bg-accent text-foreground"
                : "text-muted hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id={`panel-${activeTab}`}
        aria-labelledby={`tab-${activeTab}`}
        tabIndex={0}
      >
        {activeContent}
      </div>
    </div>
  )
}
