import dynamic from "next/dynamic"
import { Hero } from "@/components/landing/Hero"
import { WhoIsThisFor } from "@/components/landing/WhoIsThisFor"
import { ValueProps } from "@/components/landing/ValueProps"

// Dynamic below-fold components for bundle optimization
const TaskShowcase = dynamic(() => import("@/components/landing/TaskShowcase"))
const Comparison = dynamic(() => import("@/components/landing/Comparison"))
const Features = dynamic(() => import("@/components/landing/Features"))
const QuickStart = dynamic(() => import("@/components/landing/QuickStart"))
const Philosophy = dynamic(() => import("@/components/landing/Philosophy"))

export default function Home() {
  return (
    <>
      <Hero />
      <WhoIsThisFor />
      <ValueProps />
      <TaskShowcase />
      <Comparison />
      <Features />
      <QuickStart />
      <Philosophy />
    </>
  )
}
