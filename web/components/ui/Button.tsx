import { type ReactNode } from "react"
import Link from "next/link"

interface ButtonProps {
  children: ReactNode
  variant?: "primary" | "secondary" | "ghost"
  size?: "sm" | "md" | "lg"
  href?: string
  onClick?: () => void
  className?: string
  external?: boolean
}

const variants = {
  primary:
    "bg-accent text-foreground hover:bg-accent/90 shadow-lg shadow-accent/20",
  secondary:
    "bg-surface text-foreground border border-border hover:bg-surface/80",
  ghost: "text-muted hover:text-foreground hover:bg-surface",
}

const sizes = {
  sm: "px-3 py-1.5 text-sm",
  md: "px-4 py-2 text-base",
  lg: "px-6 py-3 text-lg",
}

export function Button({
  children,
  variant = "primary",
  size = "md",
  href,
  onClick,
  className = "",
  external = false,
}: ButtonProps) {
  const baseStyles =
    "inline-flex items-center justify-center rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:opacity-50 disabled:pointer-events-none"
  const combinedStyles = `${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`

  if (href) {
    if (external) {
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={combinedStyles}
        >
          {children}
        </a>
      )
    }
    return (
      <Link href={href} className={combinedStyles}>
        {children}
      </Link>
    )
  }

  return (
    <button onClick={onClick} className={combinedStyles}>
      {children}
    </button>
  )
}
