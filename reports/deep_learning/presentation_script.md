# Presentation Script — Deep Learning Image Colorization

> Course: Computer Vision · Branch: `feature/deep` · Target audience: professor + classmates
>
> Estimated duration: **13-15 min talk** + 5 min Q&A.
> Pair each slide title below with the matching figure from `reports/deep_learning/figures/`.
> Speaker notes are written as **what to say** (full sentences), not bullet outlines.

---

## Slide 1 — Title

**Show:** Project title, your name, course, branch (`feature/deep`).

**Say:**
> Good morning. I'm presenting the deep-learning track of our Multi-Methods
> Image Colorization course project. The wider project compares three families
> of colorization approaches on separate branches — scribble-based,
> example-based, and deep learning. I worked the deep-learning branch, where I
> reimplemented one classical method from scratch, fine-tuned it on COCO 2017,
> and benchmarked it against two other pretrained systems that span three
> deep-learning paradigms.

---

## Slide 2 — The problem and why it's hard

**Show:** A grayscale input with two plausible colorizations side by side
(e.g., a sky that could be blue or overcast white).

**Say:**
> Colorization is the task of predicting plausible color for a grayscale
> image. In the CIE Lab color space, that means predicting the two
> chrominance channels $a$ and $b$ from the single luminance channel $L$.
>
> The catch is that the mapping is **ill-posed and multimodal**. A gray
> sky could be blue or overcast white. An apple could be red or green. A car
> could be almost any color. There is rarely a single correct answer — only
> a *distribution* of plausible ones, and the model's design has to respect
> that fact, not pretend it doesn't exist.

---

## Slide 3 — Why classical $\ell_2$ regression fails

**Show:** Schematic of an L2 regressor producing the "desaturated mean."

**Say:**
> The most natural thing to try is to regress $(a, b)$ with an L2 loss. This
> fails in a very specific way. When the same gray sky maps to many different
> ground-truth colors in the training set, the L2 minimizer is the *mean*
> of all those colors, which is desaturated and brownish. So early regression
> CNNs produced the characteristic washed-out, sepia-tinted outputs that you
> may have seen.
>
> This is a real obstacle, and the whole story of the deep-learning paradigms
> I'll cover is how each generation tries to *avoid the desaturation trap*.

---

## Slide 4 — Three paradigms in 30 seconds

**Show:** The taxonomy table from `related_work.tex` (Table 1) — CNN, Interactive CNN, GAN with years and proxy models.

**Say:**
> Here are the three paradigms I benchmark, in chronological order:
>
> - **2016 — CNN classification.** Zhang et al. reframe the problem as
>   per-pixel classification over a quantized color palette. This is the
>   model I reimplement.
> - **2017 — Interactive CNN.** The same authors add a hint branch so users
>   can paint sparse color points. We evaluate it without hints.
> - **2020 — GAN.** The training objective becomes adversarial. DeOldify is
>   our SOTA-class anchor.
>
> A fourth paradigm — **conditional diffusion (2022 onward)** — would be
> next in the historical sequence. I'll explain in a moment why I deliberately
> excluded it from the comparison rather than substitute a degraded model.
>
> For the three I do benchmark, I'm going to walk through each architecture,
> explain what's actually *inside* the network, and tell you what each design
> choice buys at evaluation time.

---

## Slide 5 — Why diffusion is excluded (be ready, prof will ask)

**Show:** A text-only slide titled "Why diffusion is excluded" with three bullets.

**Say:**
> I want to address the diffusion question up front because it's the most
> obvious thing missing.
>
> The original project plan included a diffusion paradigm represented by
> **ControlNet conditioning a pretrained Stable Diffusion 2.1 backbone**.
> Every well-trained openly-released colorization checkpoint I could find in
> 2026 — and I checked many — is built on SD 2.1 as the image backbone.
>
> The problem: **Stability AI deprecated the entire SD 2.x line upstream.**
> The relevant Hugging Face repositories return HTTP 401 even with a valid
> token. There is no public mirror with licensed access.
>
> I tried three substitutions before deciding to exclude the paradigm
> entirely:
>
> - An **SD 1.5 brightness-ControlNet** — wrong conditioning channel,
>   produced PSNR around 6 dB, worse than a constant-gray output.
> - An **SD-InstructPix2Pix** colorization fine-tune — emitted NaN tensors and
>   required about 9 minutes per image; not a usable pipeline.
> - A **Flux-based** colorizer — 12 billion parameters; doesn't fit on the
>   6 GB GPU I used for everything else, so it would have broken the
>   "identical conditions" requirement.
>
> Reporting any of these next to faithfully-evaluated systems would have been
> **misleading**, so I exclude the paradigm and report this exclusion in the
> paper as a finding in its own right: a paradigm whose entire downstream
> ecosystem depends on a single backbone is exposed to that backbone's
> lifecycle decisions. SD 2.1's deprecation took the diffusion-colorization
> stack with it.
>
> So the rest of this talk is a **three-paradigm** comparison.

---

## Slide 6 — Common pipeline (Lab space)

**Show:** `figures/Deep_Learning_method.png`

**Say:**
> Before the per-paradigm differences, here is the structure that all three
> share. We convert the input to Lab. The model takes the L channel only
> and predicts the $ab$ channels. We then recombine $L$ with the predicted
> $ab$ and convert back to RGB.
>
> The reason Lab matters: it linearly separates *lightness* from *color*. The
> network is never asked to reproduce the image structure — that's already in
> $L$. It only predicts chrominance. Everything that follows is a different
> way of parameterizing that single $L \to (a, b)$ mapping.

---

## Slide 7 — Paradigm 1: Zhang 2016 architecture

**Show:** `figures/zhang16_arch.png` — the 8-block CNN diagram.

**Say:**
> This is the architecture I reimplemented. Eight convolutional blocks,
> 31.6 million parameters. There are three design choices that I want to
> highlight:
>
> 1. **No pooling — only stride-2 convolutions.** That means downsampling
>    is *learned*. The network chooses which features to keep at each step
>    instead of pooling them blindly. This preserves chrominance cues that
>    max-pooling tends to wash out.
>
> 2. **Dilated convolutions in the middle.** Blocks 4 through 7 hold
>    resolution constant at 32x32 and introduce dilation $d=2$ in blocks 5
>    and 6. Dilation widens the *receptive field* exponentially without
>    adding parameters or downsampling further. Why does that matter for
>    color? Because the network needs to see *enough of a region* to decide
>    "this is grass" versus "this is foliage." A wide receptive field is
>    what gives colorization its semantic coherence.
>
> 3. **A $1 \times 1$ classification head, not a regression head.** The
>    last layer outputs $Q$ logits per spatial location — a categorical
>    distribution over a *quantized* color palette. This is the key idea I'll
>    explain on the next slide.

---

## Slide 8 — Paradigm 1: classification over a quantized palette

**Show:** `figures/ab_quantization.png` — the 233 in-gamut bins.

**Say:**
> Instead of regressing two continuous numbers $(a, b)$, the $ab$ plane is
> *discretized* on a 10x10 grid and restricted to the bins that map to valid
> sRGB colors. The original paper has 313 bins from one gamut hull; my
> independently-computed grid has 233.
>
> Why classification? Because a categorical distribution over bins can
> *explicitly represent* the multimodality. If a sky could be blue or white,
> the model puts probability mass on *both* blue and white bins — and we don't
> lose either by averaging. That's the whole point.
>
> One detail: each ground-truth pixel is not a one-hot label. We use a *soft*
> encoding — Gaussian weights over the 5 nearest bins. That makes the loss
> surface smooth and lets the network express graded uncertainty.

---

## Slide 9 — Paradigm 1: class-rebalanced loss

**Show:** Equation `eq:loss` and the bin-frequency rebalancing equation `eq:rebalance`.

**Say:**
> Two more pieces complete the recipe.
>
> First, a class-rebalanced cross-entropy. Natural images are dominated by
> low-saturation colors — gray, brown, dull green. Without rebalancing, the
> network would learn to predict "everything brown" and minimize the loss
> just fine. So we reweight each pixel inversely to the frequency of its
> dominant bin in the training set, mixing with a uniform prior to avoid
> blowing up rare bins.
>
> Concretely, rare and vivid colors receive larger gradients than the
> ubiquitous low-saturation background. That's what stops the network from
> collapsing onto the easy gray solution.

---

## Slide 10 — Paradigm 1: annealed-mean decoding

**Show:** Equation `eq:annealed`.

**Say:**
> At inference we still need a single $(a, b)$ pair per pixel, so we have to
> collapse the distribution to a point. Two naive options:
>
> - Take the **mode** — you get vivid but spatially inconsistent outputs,
>   because nearby pixels can jump to different bins.
> - Take the **mean** — you get spatially smooth but desaturated outputs.
>
> The trick is to *temper* the distribution with a temperature $T$ before
> taking the mean. As $T$ goes to 0 we recover the mode; as $T$ goes to 1 we
> recover the mean. We use $T = 0.38$, which is the sweet spot between the
> two failure modes. This single hyperparameter is what trades saturation
> against spatial coherence in the final output.

---

## Slide 11 — Paradigm 2: Zhang 2017 (interactive CNN)

**Show:** `figures/zhang17_arch.png`

**Say:**
> Zhang's 2017 paper attacks a different problem: *interactive* colorization.
> The user paints sparse color points and the network propagates them
> globally, in real time.
>
> The architecture extends the 2016 backbone in two ways. First, **U-shape
> skip connections** — those purple arrows. Sharp boundaries from the
> encoder flow straight into the decoder, which is what lets a single
> clicked color point reach distant pixels along sharp edges. Second, a
> **second branch** that encodes the user hints together with a binary mask
> and concatenates them at the bottleneck.
>
> We evaluate it in **automatic mode** — all hints set to zero — to put it
> on equal footing with the other paradigms. And this is the key thing to
> say to a professor: the model was *not* optimized for that regime. It is
> a hint-driven network deprived of hints. So when we see it underperform
> on the benchmark, that is not a failure of the architecture — it's a
> finding about which tool to pick when.

---

## Slide 12 — Paradigm 3: DeOldify (GAN)

**Show:** `figures/deoldify_arch.png`

**Say:**
> DeOldify is our SOTA-class anchor. There's no peer-reviewed paper, but the
> recipe is well-documented. Three components:
>
> 1. **U-Net generator with a ResNet-34 encoder** that is initialized from
>    ImageNet. Why ImageNet? Because it imports strong mid-level features
>    — textures, parts, object semantics — for free. The network already
>    knows what a fire truck looks like before it ever sees a colorization
>    example.
>
> 2. **Self-attention at the bottleneck.** A plain convolutional bottleneck
>    can only relate spatially nearby features. Self-attention lets a single
>    airplane pixel attend to the *entire* sky and adopt a globally
>    consistent color. This is the mechanism behind DeOldify's confident,
>    coherent saturation on large ambiguous regions.
>
> 3. **NoGAN training.** Classical colorization GANs are unstable. NoGAN
>    pretrains $G$ alone with a perceptual loss, then pretrains $D$ alone on
>    $G$'s frozen outputs, and only then runs a *very short* adversarial
>    fine-tune — about 1-3% of total epochs. The critic never has time to
>    overwhelm the generator. That stability is what makes DeOldify usable
>    in production.

---

## Slide 13 — Experimental setup (briefly)

**Show:** A simple text slide with the key facts.

**Say:**
> One paragraph on the setup:
>
> - **Dataset:** COCO 2017. We fine-tune on `train2017`; we use `val2017`
>   for best-checkpoint selection during training.
> - **Benchmark:** a fixed stratified 1,000-image subset of `test2017` with
>   seed 42. Importantly, we do *not* evaluate on `val2017` — that would
>   leak, because the training procedure uses it for checkpoint selection.
>   This is the leakage policy our `docs/benchmark_methodology.md` documents.
> - **Uncertainty:** every reported metric is `mean [95% CI]` via a
>   10,000-resample percentile bootstrap, also seed 42. Single numbers
>   hide sampling variability; bootstrap intervals do not.
> - **Hardware:** GTX 1660 SUPER, 6 GB. We trained for 8 epochs in fp32,
>   batch size 4. (We tried mixed precision; it NaN-ed mid-epoch, which is a
>   real lesson about why AMP is not free.)

---

## Slide 14 — Training curves

**Show:** `figures/training_curves.png`

**Say:**
> The training curves for our fine-tuning. We trained in two phases — five
> batch-capped epochs first, then three full-data epochs resumed from the
> prior checkpoint. By epoch 8 validation PSNR climbs to about 23.6 dB. The
> two validation-loss spikes you can see (epochs 2 and 7) are pathological
> batches and have been masked off the plotted line, which is annotated.

---

## Slide 15 — Quantitative results

**Show:** `figures/metrics_comparison.png` + the results table (`tab:results` in `results.tex`).

**Say:**
> Here is the headline comparison on the 1,000-image benchmark, all numbers
> as mean and 95% bootstrap CI.
>
> - **Our fine-tuned Zhang16: PSNR 23.29, SSIM 0.919, LPIPS 0.192.**
> - **DeOldify: PSNR 24.00, SSIM 0.919, LPIPS 0.148.**
> - **Zhang17 in auto mode: PSNR 18.82.** Substantially lower, for the
>   reasons I explained.
> - **Pretrained Zhang16: PSNR 17.14** — but **this number is misleading**,
>   and I want to be honest about it. The official ECCV head ships 313 bins
>   from a different gamut. My head has 233 bins. They are
>   dimensionally incompatible, so this row measures "ECCV encoder + my
>   untrained 233-bin head" — not the real pretrained model. The fix would
>   be to re-derive the 313-bin gamut hull; I document this as future work.
>
> So the substantive finding is: **our fine-tuned model closes most of the
> gap to the GAN SOTA anchor while running about 2.8 times faster.**

---

## Slide 16 — Qualitative comparison

**Show:** `figures/qualitative_grid.png`

**Say:**
> The qualitative grid. Input, our pretrained baseline, our fine-tuned
> model, Zhang17 in auto, DeOldify, then ground truth.
>
> Two things to point out:
>
> - The **pretrained** column is the head-mismatch artifact I just
>   described — the characteristic grid-pattern color noise from an
>   untrained 233-bin head. Fine-tuning eliminates it completely.
> - **Zhang17 in auto mode** shows the failure mode of a hint-driven
>   network without hints: saturated blotches and color bleeding across
>   object boundaries.
> - **Our fine-tuned model and DeOldify** both produce coherent,
>   naturally-colored results. DeOldify is the more saturated of the two;
>   we are slightly more conservative on large ambiguous regions.

---

## Slide 17 — Per-image strengths

**Show:** `figures/model_strengths.png` + the win-rate table.

**Say:**
> Aggregate numbers can hide a lot, so I drilled into per-image win rates.
> A few takeaways:
>
> - DeOldify wins **PSNR on 58%** of images and **LPIPS on 86%** of
>   images.
> - **We win SSIM on nearly two-thirds** — about 64% — of images, because
>   our model is more structurally faithful on conservative,
>   low-saturation content.
> - Our biggest single-image win is on a desaturated outdoor scene,
>   *plus 15 dB* over the runner-up: our conservative behavior matches the
>   ground truth and DeOldify's adversarial confidence overshoots.
> - DeOldify's biggest single-image win is on an airplane against a
>   saturated blue sky, *plus 14 dB*: the opposite regime, where DeOldify
>   commits boldly and we hedge gray.
>
> So each model has a niche, and the aggregate numbers are averages over
> those niches.

---

## Slide 18 — Speed-quality frontier (discussion)

**Show:** Text bullets — PSNR vs. inference time.

**Say:**
> The three evaluated points occupy distinct positions on the speed-quality
> frontier:
>
> - **CNNs are single forward passes** — 0.17 to 0.22 seconds per image even
>   on a 6 GB GPU. Our model at 0.18s, DeOldify at 0.51s.
> - **DeOldify buys its perceptual edge with a heavier U-Net and a larger
>   render resolution.** Roughly 2.8x slower than us.
>
> So the practical recommendation is paradigm-dependent:
>
> - For **automatic, latency-sensitive** colorization, a well-trained CNN is
>   the rational default.
> - For **maximum perceptual quality**, use a GAN like DeOldify and accept
>   the latency.
> - For **human-in-the-loop control**, use an interactive CNN *with* hints;
>   do not expect it to work unguided.

---

## Slide 19 — Limitations and future work

**Show:** Bullet list.

**Say:**
> Two honest limitations I want to surface:
>
> 1. The pretrained-baseline row is **not** a faithful ECCV baseline because
>    of the head-mismatch. To fix it we would re-derive the 313-bin gamut
>    hull and load the official head.
> 2. The benchmark is 1,000 images on a single GPU with no human-preference
>    study. Perceptual quality is proxied by LPIPS, not measured directly.
>
> Beyond those, the diffusion paradigm should be revisited once a maintained
> colorization checkpoint exists that does not depend on the deprecated
> Stable Diffusion 2.1 backbone — that would complete the four-paradigm
> picture with a measured rather than inferred frontier point.

---

## Slide 20 — Conclusion

**Show:** Three bullet recommendations.

**Say:**
> To wrap up:
>
> 1. I reimplemented the Zhang 2016 classification CNN, fine-tuned it on
>    COCO 2017, and reached PSNR 23.29 dB, SSIM 0.919, LPIPS 0.192 on a
>    leakage-controlled 1,000-image `test2017` benchmark.
> 2. Our model matches DeOldify on structural similarity, trails it
>    modestly on pixel fidelity and perceptual quality, and beats it by
>    roughly 2.8x on inference speed.
> 3. The methodology — class-rebalanced classification, soft palette
>    quantization, annealed-mean decoding — is what actually does the
>    work. Each piece exists to dodge the desaturation trap.
> 4. The diffusion paradigm was scoped in and then explicitly excluded once
>    its only viable backbone (SD 2.1) was deprecated upstream — itself a
>    reproducibility finding rather than an evaluation result.
>
> Thank you. I'm happy to take questions.

---

## Q&A Preparation

These are the questions a senior CV professor is most likely to ask. Drill them
beforehand so you can answer in 30-60 seconds each.

**Q: Why didn't you just use a different diffusion model?**
A: I tried three — SD 1.5 brightness-ControlNet, SD-InstructPix2Pix, and
Flux. The first produced PSNR around 6 dB because the conditioning channel was
wrong. The second emitted NaNs at 9 minutes per image. The third doesn't fit
on a 6 GB consumer GPU, which would break the "identical conditions"
constraint of the benchmark. None of them is a faithful diffusion-colorization
baseline; reporting one would have been misleading.

**Q: Why didn't you just use the official 313-bin head?**
A: Dimensional incompatibility. Our gamut hull yields 233 in-gamut bins;
the official one yields 313 from a different hull. You can't load a 313-channel
1x1 conv into a 233-channel slot without retraining the head. We documented this
and only report the *fine-tuned* row as a faithful colorization result.

**Q: Why class-rebalanced cross-entropy and not focal loss?**
A: They serve similar purposes — both upweight rare-but-hard examples — but
class-rebalanced CE is the recipe in the source paper and we wanted a faithful
reimplementation. Focal loss is a reasonable replacement to try; we did not.

**Q: What is the receptive field of your network in pixels?**
A: With three stride-2 downsamplings and dilation-2 in the middle blocks, the
effective receptive field at the classification head covers roughly the full
input image at 256x256. That's by design: colorization is semantic, so we
want every output pixel to "see" enough of the scene to commit to a color.

**Q: Why temperature 0.38? Did you tune it?**
A: It's the value the original paper found best on ImageNet; we kept it
without retuning. A temperature sweep on the val set would be a clean
follow-up experiment.

**Q: Could you replace the classification head with a discrete diffusion
head?**
A: Yes, that's a natural extension — sample bins iteratively. It would buy
diversity at inference (multiple plausible colorizations) at the cost of
latency. Out of scope here.

**Q: How did you handle the AMP NaN?**
A: We disabled mixed precision and ran in fp32. The fp16 overflow happened in
the loss term at iteration 1649; rather than chase a specific numerical
guard, we accepted the slower fp32 path. Batch size 4 still fits the 6 GB
card.

**Q: Is the win-rate analysis statistically significant?**
A: The aggregate metric CIs are disjoint between our model and DeOldify on
LPIPS and on PSNR; they overlap on SSIM. The per-image win rates are
descriptive — they don't carry a separate significance test.

**Q: What's the failure case nobody else mentions?**
A: Semantically ambiguous content that none of the models handle well — a
shirt that could be any color, or skin with unusual lighting. All three
paradigms default to the most-frequent training color in those cases. This
is *fundamental to the task*, not specific to any architecture.

---

## Presenter checklist (before walking in)

- [ ] Slide deck exported from `reports/deep_learning/figures/` PNGs
- [ ] `Report.pdf` open in a second tab in case the prof wants a deep cite
- [ ] Notebook `notebooks/03_*` open on a colorized image, as a live demo if asked
- [ ] Memorize: 23.29 / 0.919 / 0.192 (our numbers), 24.00 / 0.919 / 0.148 (DeOldify), 1000 images / seed 42 / 10000 bootstrap
- [ ] Practice the answer to the "head mismatch" question — it WILL come up
- [ ] Practice the answer to "why no diffusion" — the SD 2.1 deprecation chain
