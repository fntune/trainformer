import { CodeBlock } from "@/components/ui/CodeBlock"

const quickStartCode = `from trainformer import Trainer
from trainformer.tasks import ImageClassification
from trainformer.callbacks import ModelCheckpoint
from trainformer.loggers import WandbLogger

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
).fit()`

export default function QuickStart() {
  return (
    <section className="py-20">
      <div className="max-w-4xl mx-auto px-4">
        <h2 className="text-3xl font-bold text-foreground text-center mb-4">
          Get started in seconds
        </h2>
        <p className="text-muted text-center mb-8 max-w-2xl mx-auto">
          Install trainformer and start training immediately.
          No configuration files, no boilerplate.
        </p>

        <div className="space-y-6">
          <div className="flex items-center justify-center">
            <div className="flex items-center gap-2 px-6 py-3 rounded-lg bg-surface border border-border">
              <span className="text-accent font-mono">$</span>
              <code className="font-mono text-foreground">
                pip install trainformer
              </code>
            </div>
          </div>

          <CodeBlock code={quickStartCode} lang="python" title="train.py" />
        </div>
      </div>
    </section>
  )
}
