# Trainformer Landing Page Plan

## Core Positioning

**Target**: ML researchers and applied engineers who want to iterate fast.

**Key Differentiators**:
- **Experiment-first** - Built for rapid iteration, not production boilerplate
- **Debuggable** - Step through training with a debugger, not logs
- **Pluggable** - Swap components without rewriting pipelines
- **Power usage** - Full control when you need it, sensible defaults when you don't

---

## Core Messaging

**Headline**: "The Experiment Framework for Deep Learning"
**Subhead**: Iterate fast. Debug everything. Ship when ready.

---

## Page Structure

### 1. Hero Section

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│        trainformer                                          │
│                                                             │
│   The Experiment Framework for Deep Learning                │
│   Iterate fast. Debug everything. Ship when ready.          │
│                                                             │
│   [pip install trainformer]  [GitHub]                       │
│                                                             │
│   ┌───────────────────────────────────────────────────┐    │
│   │  from trainformer import Trainer                   │    │
│   │  from trainformer.tasks import ImageClassification │    │
│   │                                                    │    │
│   │  task = ImageClassification("resnet50")            │    │
│   │  Trainer(task, data="cifar10", epochs=10).fit()    │    │
│   └───────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Code example**: 4-5 lines showing complete training setup

---

### 2. Value Props (4 pillars)

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Experiment     │  │   Debuggable    │  │   Pluggable     │  │   Powerful      │
│    First        │  │                 │  │                 │  │                 │
│                 │  │ Step through    │  │ Swap backbones, │  │ AMP, compile,   │
│ Iterate in      │  │ training code   │  │ losses, heads   │  │ distributed,    │
│ minutes, not    │  │ with a real     │  │ without         │  │ LoRA - when     │
│ hours           │  │ debugger        │  │ rewrites        │  │ you need it     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

### 3. Tasks Showcase

Tabbed interface showing supported domains:

**Vision Tab**:
```python
from trainformer.tasks import MetricLearning

task = MetricLearning(
    backbone="efficientnet_b0",
    loss="arcface",
    embedding_dim=512,
)
Trainer(task, data="stanford_products").fit()
```
- ImageClassification
- MetricLearning (ArcFace, CosFace)
- SSL (SimCLR, MoCo, DINO, MAE)

**NLP Tab**:
```python
from trainformer.tasks import CausalLM

task = CausalLM("meta-llama/Llama-2-7b", adapter=LoRA(r=8))
Trainer(task, data="databricks/dolly").fit()
```
- CausalLM (GPT, Llama, Mistral)
- Seq2Seq (T5, BART)
- MaskedLM (BERT)

**Multimodal Tab**:
```python
from trainformer.tasks import VLM

task = VLM("llava-1.5-7b")
Trainer(task, data="coco_captions").fit()
```
- CLIP
- VLM (LLaVA, Qwen-VL)

---

### 4. Comparison Table

| Feature | trainformer | PyTorch Lightning | HF Trainer |
|---------|:-----------:|:-----------------:|:----------:|
| Config format | Python | Python | YAML + Python |
| Vision tasks | ✓ | manual | limited |
| NLP tasks | ✓ | manual | ✓ |
| Multimodal | ✓ | manual | limited |
| LoRA built-in | ✓ | ✗ | ✓ |
| Learning curve | low | medium | high |
| Core lines | ~600 | ~5000+ | ~10000+ |

---

### 5. Features Grid

```
┌───────────────────┬───────────────────┬───────────────────┐
│  Mixed Precision  │  torch.compile    │  Distributed      │
│  fp16 / bf16      │  2x speedup       │  via Accelerate   │
├───────────────────┼───────────────────┼───────────────────┤
│  Callbacks        │  Logging          │  Evaluation       │
│  Checkpoint, EMA  │  WandB, TB, MLflow│  FAISS, KNN       │
├───────────────────┼───────────────────┼───────────────────┤
│  Adapters         │  Gradient         │  Export           │
│  LoRA, QLoRA      │  Accumulation     │  ONNX, TorchScript│
└───────────────────┴───────────────────┴───────────────────┘
```

---

### 6. Quick Start

```bash
pip install trainformer
```

```python
from trainformer import Trainer
from trainformer.tasks import ImageClassification

# Config is code - type-safe, IDE-friendly
task = ImageClassification(
    model_name="resnet50",
    num_classes=10,
    pretrained=True,
)

# Full-featured training in one line
Trainer(
    task,
    data="cifar10",
    epochs=100,
    lr=1e-4,
    precision="fp16",
    callbacks=[ModelCheckpoint("best.pth")],
    logger=WandbLogger(project="my-project"),
).fit()
```

---

### 7. Philosophy Section

> **"Your experiment, your rules"**
>
> trainformer gives you the iteration speed of notebooks with
> the code quality of production systems.

Three principles:
1. **Debuggable by default** - Step through any line, inspect any tensor
2. **Pluggable architecture** - Swap components without rewriting
3. **Power when needed** - Sensible defaults, full control available

---

### 8. Footer

- GitHub link
- Documentation link
- `pip install trainformer`
- Version badge (v0.1.0)

---

## Technical Implementation Notes

**Stack**: Next.js + Tailwind CSS (aligns with existing project patterns)

**Key UI Elements**:
- Syntax-highlighted code blocks (Shiki or Prism)
- Dark mode support (developer audience)
- Tabbed interface for task domains
- Animated feature comparison table
- Copy-to-clipboard on code blocks

**Responsive**: Mobile-first, code blocks scroll horizontally on small screens

---

## Project Structure (Future-Proof)

```
web/                          # Standalone Next.js app
├── app/
│   ├── (marketing)/          # Marketing pages group
│   │   ├── page.tsx          # Landing page (/)
│   │   ├── pricing/          # Future: /pricing
│   │   └── about/            # Future: /about
│   ├── docs/                 # Future: documentation
│   │   └── [...slug]/        # Dynamic doc routes
│   ├── blog/                 # Future: blog posts
│   │   └── [slug]/
│   ├── layout.tsx            # Root layout
│   └── globals.css
├── components/
│   ├── landing/              # Landing page components
│   │   ├── Hero.tsx
│   │   ├── ValueProps.tsx
│   │   ├── TaskShowcase.tsx
│   │   ├── Comparison.tsx
│   │   ├── Features.tsx
│   │   ├── QuickStart.tsx
│   │   └── Philosophy.tsx
│   ├── ui/                   # Shared UI components
│   │   ├── CodeBlock.tsx
│   │   ├── Button.tsx
│   │   ├── Tabs.tsx
│   │   └── CopyButton.tsx
│   └── layout/               # Layout components
│       ├── Header.tsx
│       └── Footer.tsx
├── lib/
│   └── syntax-highlight.ts   # Shiki config
├── public/
│   └── og-image.png          # Social preview
├── package.json
├── tailwind.config.ts
├── next.config.ts
└── tsconfig.json
```

**Why this structure**:
- `(marketing)` route group keeps landing pages organized
- Placeholder routes for future docs/blog expansion
- Shared `components/ui/` for design system consistency
- `lib/` for utilities that grow with the site

---

## Implementation Steps

### Phase 1: Project Setup
1. Initialize Next.js app in `web/` with TypeScript, Tailwind, App Router
2. Configure Tailwind with dark mode, custom colors
3. Set up Shiki for syntax highlighting

### Phase 2: Core Components
1. `CodeBlock.tsx` - Syntax highlighting with copy button
2. `Button.tsx` - Primary/secondary variants
3. `Tabs.tsx` - For task showcase section
4. `Header.tsx` / `Footer.tsx` - Site navigation

### Phase 3: Landing Page Sections
1. `Hero.tsx` - Headline, subhead, CTA buttons, code snippet
2. `ValueProps.tsx` - 3 pillars grid
3. `TaskShowcase.tsx` - Tabbed code examples (Vision/NLP/Multimodal)
4. `Comparison.tsx` - Framework comparison table
5. `Features.tsx` - Feature grid (6 features)
6. `QuickStart.tsx` - Install command + full example
7. `Philosophy.tsx` - Design principles quote

### Phase 4: Polish
1. Responsive design verification
2. Dark mode styling
3. OG image for social sharing
4. SEO metadata

---

## Dependencies

```json
{
  "dependencies": {
    "next": "^15",
    "react": "^19",
    "react-dom": "^19",
    "shiki": "^1.0"
  },
  "devDependencies": {
    "typescript": "^5",
    "tailwindcss": "^4",
    "@tailwindcss/typography": "^0.5"
  }
}
```

---

## Web Interface Guidelines Checklist

### Accessibility
- [ ] Skip link in layout.tsx (`<a href="#main">Skip to content</a>`)
- [ ] `aria-label` on icon buttons (GitHub, Copy)
- [ ] Keyboard navigation for Tabs (arrow keys via `onKeyDown`)
- [ ] `focus-visible:ring-*` on all interactive elements
- [ ] Heading hierarchy: single `<h1>`, descending `<h2>`-`<h6>`
- [ ] `alt` text on all images, `aria-hidden="true"` on decorative icons

### Animation & Motion
- [ ] Honor `prefers-reduced-motion` for all animations
- [ ] Animate only `transform`/`opacity` - never `transition: all`
- [ ] Animations must be interruptible

### Typography
- [ ] Use curly quotes `"` `"` not straight quotes
- [ ] Use `…` not `...` in loading states
- [ ] `text-wrap: balance` on headings
- [ ] `font-variant-numeric: tabular-nums` for code line numbers

### Dark Mode
- [ ] `color-scheme: dark` on `<html>`
- [ ] `<meta name="theme-color">` matching background
- [ ] Explicit `background-color` and `color` on native `<select>`

### Navigation & State
- [ ] Tabs update URL (`?tab=vision`) for deep-linking
- [ ] Use `<a>`/`<Link>` for navigation, `<button>` for actions

### Touch & Mobile
- [ ] `touch-action: manipulation` on buttons
- [ ] Code blocks: `overflow-x-auto` for horizontal scroll
- [ ] Safe areas: `env(safe-area-inset-*)` for full-bleed

---

## Vercel React Best Practices Checklist

### Bundle Size (CRITICAL)
- [ ] `bundle-dynamic-imports`: Lazy-load Shiki highlighter with `next/dynamic`
- [ ] `bundle-defer-third-party`: Load analytics after hydration
- [ ] `bundle-barrel-imports`: Import directly, avoid barrel files
- [ ] `bundle-preload`: Preload tab content on hover/focus

### Server Performance (HIGH)
- [ ] `server-serialization`: Minimize data passed to client components
- [ ] Use Server Components for static sections (Hero, Philosophy)

### Rendering (MEDIUM)
- [ ] `rendering-content-visibility`: Apply to below-fold sections
- [ ] `rendering-hoist-jsx`: Extract static JSX outside components
- [ ] `rendering-conditional-render`: Use ternary, not `&&`

### Re-renders (MEDIUM)
- [ ] `rerender-lazy-state-init`: Pass function to `useState` for tabs
- [ ] `rerender-memo`: Memoize `CodeBlock` component

---

## Implementation Patterns

### CodeBlock with Dynamic Shiki

```tsx
// components/ui/CodeBlock.tsx
import dynamic from "next/dynamic"

const HighlightedCode = dynamic(
  () => import("./HighlightedCode"),
  { ssr: false, loading: () => <pre className="animate-pulse" /> }
)

export function CodeBlock({ code, lang }: Props) {
  return <HighlightedCode code={code} lang={lang} />
}
```

### Tabs with URL State

```tsx
// components/landing/TaskShowcase.tsx
"use client"
import { useSearchParams, useRouter } from "next/navigation"

export function TaskShowcase() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const tab = searchParams.get("tab") ?? "vision"

  const setTab = (value: string) => {
    router.replace(`?tab=${value}`, { scroll: false })
  }

  return (
    <div role="tablist" aria-label="Task domains">
      {/* tabs with onKeyDown for arrow navigation */}
    </div>
  )
}
```

### Below-fold Lazy Loading

```tsx
// app/(marketing)/page.tsx
import dynamic from "next/dynamic"

// Static above-fold
import { Hero } from "@/components/landing/Hero"
import { ValueProps } from "@/components/landing/ValueProps"

// Dynamic below-fold
const TaskShowcase = dynamic(() => import("@/components/landing/TaskShowcase"))
const Comparison = dynamic(() => import("@/components/landing/Comparison"))
const Features = dynamic(() => import("@/components/landing/Features"))
```
