import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Header } from "@/components/layout/Header"
import { Footer } from "@/components/layout/Footer"
import "./globals.css"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "trainformer — The Experiment Framework for Deep Learning",
  description:
    "Iterate fast. Debug everything. Ship when ready. A Python-first training library for vision, NLP, and multimodal deep learning.",
  keywords: [
    "deep learning",
    "machine learning",
    "pytorch",
    "training",
    "vision",
    "nlp",
    "multimodal",
    "python",
  ],
  authors: [{ name: "trainformer" }],
  openGraph: {
    title: "trainformer — The Experiment Framework for Deep Learning",
    description:
      "Iterate fast. Debug everything. Ship when ready. A Python-first training library for vision, NLP, and multimodal deep learning.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "trainformer — The Experiment Framework for Deep Learning",
    description:
      "Iterate fast. Debug everything. Ship when ready. A Python-first training library for vision, NLP, and multimodal deep learning.",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="theme-color" content="#0a0a0a" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <Header />
        <main id="main">{children}</main>
        <Footer />
      </body>
    </html>
  )
}
