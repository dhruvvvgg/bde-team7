# TGT-M explained from zero: the idea, the architecture, and every line of the code

This guide explains the proposed model, TGT-M, as if you had never seen a neural network. It goes slowly, uses small examples, and always points to the real file and function in the project, so that after reading it you can open the code and follow it, and answer questions about it in the review. Every number in this guide (shapes, parameter counts, settings) was read from the code or computed by running it.

How to read it: Part 1 and 2 explain the problem and the data the model receives. Part 3 teaches only the neural-network ideas you need. Part 4 explains each building block of TGT-M on its own. Part 5 puts the blocks together, stage by stage, with the exact shape of the data at every step. Part 6 explains how the model is trained. Part 7 shows how the output becomes an answer. Part 8 explains the other models we compared with. Part 9 is the honest summary of what the model achieved. Part 10 is a map of the files and a list of questions you may be asked.

---

## Part 1. The problem, in plain words

### 1.1 What is a well reading?

Groundwater is water under the ground. India measures it by lowering a tape into observation wells and noting how far below the ground the water surface is. This number is called the **depth to groundwater**, in metres below ground level. A reading of 3 metres means the water is 3 metres down. A deeper reading (a bigger number) means there is less water: the well is more stressed.

We have 2,759 wells in 19 states. Each well is measured four times a year: in January, May, August and November. We call each of these a **round**. From 2000 to 2022 there are 92 rounds. If every reading existed we would have 2,759 times 92 = 253,828 readings, but 13.6 percent are missing (a person did not visit, or the record was lost). So every well has a diary with some pages missing.

### 1.2 What do we want to predict?

We want to estimate how stressed a well is in a given round. But a depth of 8 metres is normal for one well and an emergency for another. So we do not judge the raw depth. We judge how unusual the reading is compared with the same well's own history for the same season.

The recipe, which the preprocessing code follows exactly:

1. Take the earlier readings of this well in the same season (for example all the earlier May readings). We need at least 5.
2. Compute their average. This is the **baseline mean**.
3. Compute how much they usually vary. This is the **baseline standard deviation** (SD). We never let the SD be smaller than 0.10 metres, so that a very steady well does not create absurd scores.
4. The **anomaly score z** of a reading is how many SDs it sits above the baseline mean:

   z = (reading − baseline mean) / SD

5. We clip z to the range −5 to +5 so that a freak reading cannot dominate.

A worked example from the real data: the well at Cherfully, Thane, round May 2006. The five earlier May readings give a baseline mean of 1.684 m and an SD of 0.376 m. The May 2006 reading is deeper than usual, and its z is 2.037. The stress classes are decided by z:

| z | class |
|---|---|
| below 1 | Normal |
| 1 to 1.5 | Watch |
| 1.5 to 2 | Moderate Stress |
| 2 or more | High Stress |

So z = 2.037 means High Stress. You can check this by hand in the notebook (section 5), and the pipeline's stored value is identical.

The word **past-only** is the most important idea in the project. The baseline for a round uses only readings from earlier rounds. It never touches the current round or any later round. If it did, the model could "peek at the answer" and look better than it really is. This peeking is called **leakage**, and we built tests that try to catch it.

### 1.3 What exactly does TGT-M predict?

TGT-M predicts the anomaly score **z** of the current round, while the current reading itself is hidden from the model. It is allowed to see:

- the earlier rounds (the last 12 rounds, which is three years),
- the season and the rainfall of the current round (rainfall that ended the month before the round, which is already known),
- facts about the well that do not change (well type, aquifer type, well depth, and so on),
- summaries of the past (previous readings, the baseline, how much it varied).

From these it must guess how far above or below normal this round is. Once we have a guess for z, we can convert it back into a depth: depth = baseline mean + z × SD. And from z we get the stress class using the table above.

Why predict z and not the depth directly? Because z is comparable across wells. A model that predicts z does not have to learn that Madhya Pradesh wells are deeper than Kerala wells; the baseline already took care of it.

---

## Part 2. What the model receives

The model receives two things for every row it has to predict. Think of them as a **diary** (the sequence) and a **fact sheet** (the static vector).

The code that builds both is in `modeling/data.py`, function `build()`.

### 2.1 The diary: a table of 13 rows and 10 columns

For the round we want to predict (call it round t) we write down 13 steps: the 12 rounds before it (t−12 to t−1) and the round t itself. That is 13 rows. Each row has 10 numbers, called **channels**. In code the variable is `seq` and its shape is (number of rows, 13, 10). The constant `L = 12` in `data.py` is the number of past rounds.

The ten channels:

| channel | name | what it holds |
|---|---|---|
| 0 | relative reading | the well's reading in that round, expressed as a z-like number: (reading − baseline mean of round t) / SD, clipped to ±10. If the reading is missing it is 0 |
| 1 | mask | 1 if the reading was observed, 0 if it is missing or hidden |
| 2 | season January | 1 if this step is a January round |
| 3 | season May | 1 if a May round |
| 4 | season August | 1 if an August round |
| 5 | season November | 1 if a November round |
| 6 | valid | 1 if this round exists, 0 if we have gone back before the table starts (padding) |
| 7 | log rain 1 month | the natural log of (1 + rainfall of the month before that round, in mm) |
| 8 | log rain 3 months | the same for the 3 months before that round |
| 9 | spare | always 0, kept for later |

Three things to understand about this table:

**The current reading is hidden.** In the last row (step t) channel 0 is 0 and the mask is 0, always. Only the season and rainfall of round t are filled. This is the "hidden reading" that the model has to estimate. The code line is `val[:, -1] = np.nan  # current reading hidden`.

**Missing readings are kept as gaps, not filled in.** If the reading at step t−5 does not exist, channel 0 is 0 and the mask is 0. We do not invent a value. The model is told that a value is missing through the mask. This is the reason we use GRU-D (Part 4.6), a network designed for exactly this.

**Every depth is expressed relative to round t's baseline.** All twelve past readings are converted to "how far above or below the baseline mean of round t". This makes the numbers comparable between wells.

In Model A (no rainfall), channels 7 and 8 are set to zero. In Model B they hold the rainfall. That is the only difference in the diary between Model A and Model B.

### 2.2 The fact sheet: the static vector

Besides the diary, the model receives one vector of numbers per row: the **static vector**. In code it is `X` (and after scaling, `B.X`). It has 88 numbers for Model B and 55 numbers for Model A. The pieces are:

- 48 numeric columns from the preprocessing table (class A columns only, which are the ones that are safe to use), of which 24 are rainfall features (1, 3, 6 and 12 month rainfall totals, monsoon-year-to-date rain, and so on);
- 6 depth lags re-expressed relative to the baseline (previous round, last observed, same season last year, same season two years ago, maximum to date, minimum to date), each computed as (value − baseline mean) / SD and clipped to ±10;
- 12 one-hot numbers for three categorical columns: season, well type, aquifer type (one-hot means one 0/1 number per possible category);
- 22 **missing flags** (13 in Model A): for a column that sometimes has missing values we add an extra 0/1 number that says "this value was missing here". This is done by the `Scaler` class.

48 + 6 + 12 = 66 numbers, plus 22 flags = 88 in Model B. Without the 24 rainfall columns and their flags the count becomes 42 + 13 = 55 for Model A.

What is deliberately **not** in the static vector: latitude, longitude, state, district and the year. We removed them (the set `DROP` in `data.py`) so that the model cannot recognise a held-out state or a future year by its name or place. It has to learn from the behaviour of the water, not from the address.

### 2.3 Scaling

Neural networks learn better when the numbers are of similar size. The `Scaler` class in `data.py` does three things, and it learns them from the **training rows only** (never from validation or test rows, to avoid leakage):

1. Fill missing values with the median of the column (the middle value).
2. Subtract the mean and divide by the standard deviation (standardise), then clip to ±8.
3. Add the missing flags described above.

The class `Bundle` in `modeling/lib.py` applies this scaler and also standardises the two rainfall channels of the diary. For Model A it removes the rainfall columns and zeroes the rainfall channels.

### 2.4 How many rows, and how we split them

After requiring a reading to exist, at least 5 earlier same-season readings, and a round that is not almost empty, we have 160,580 usable rows. We split by **time**, never randomly:

- training: 2000 to 2015, 104,437 rows (the model learns from these),
- validation: 2016 and 2017, 15,513 rows (we choose settings with these),
- test: 2018 to 2022, 40,630 rows (the final exam, touched once).

Splitting by time is honest because in real life you always predict the future from the past. A random split would let the model learn from the future and cheat.

---

## Part 3. The few neural-network ideas you need

### 3.1 A neuron, a layer, a network

A **neuron** takes some numbers, multiplies each by a **weight** (how important that input is), adds them up, adds a **bias** (a constant shift), and passes the result through an **activation function** (a bend that lets the network learn curves instead of only straight lines). A **layer** is many neurons side by side. A **network** is layers one after another. In PyTorch a layer of this kind is `nn.Linear(inputs, outputs)`: it holds a matrix of weights and a vector of biases.

The weights and biases are called the **parameters**. TGT-M has 75,646 of them in its Model B setting. At the start they are random. Training adjusts them.

### 3.2 Training = guess, measure, correct, repeat

1. Show the network some examples and let it guess.
2. Measure how wrong it was with a **loss** (a single number, bigger means worse).
3. Work out, for every parameter, which way to nudge it to make the loss smaller. This is the **gradient**, computed automatically by `loss.backward()`.
4. Nudge all parameters a small step in that direction. The size of the step is the **learning rate**. The routine that does the nudging is the **optimiser**; we use **AdamW**.
5. Repeat with the next group of examples. A group is a **batch** (we use 1,024 rows). One pass over all the training rows is an **epoch**.

### 3.3 Overfitting and how we fight it

A network can learn the training rows by heart and then do badly on new rows. This is **overfitting**. We fight it with:

- **Dropout**: during training, randomly switch off a share of the numbers inside the network (for example 30 percent), so it cannot rely on any single path.
- **Weight decay**: a small pull that keeps weights small (the `wd` setting of AdamW).
- **Early stopping**: after every epoch we measure the error on the validation rows. If it does not improve for 3 epochs in a row, we stop and go back to the best version. The validation rows are never used to adjust weights.
- **Gradient clipping**: if the nudge is huge, shrink it (`clip_grad_norm_(..., 1.0)`), so one bad batch cannot wreck the network.

### 3.4 Words you will see in the code

- **Tensor**: a block of numbers with a shape, like a table (2 dimensions) or a stack of tables (3 dimensions). Shape (8, 13, 10) means 8 rows, each a table of 13 by 10.
- **Batch dimension**: the first number of the shape. The network processes many rows at once.
- **Hidden size** `d` (or `hidden`): the width of the network's internal representation. TGT-M uses 48 for Model B and 32 for Model A.
- **Seed**: the starting point of the random numbers. We train with 3 seeds and average, to show the result is not luck.

---

## Part 4. The building blocks of TGT-M, one at a time

TGT-M is made by joining pieces taken from two published models: **GRU-D** (Che et al., 2018) and the **Temporal Fusion Transformer, TFT** (Lim et al., 2021). Here is each piece with a small intuition, and its code.

### 4.1 Activation functions: ELU, sigmoid, softmax

- **ELU** is a gentle bend: positive numbers pass unchanged, negative numbers are squashed toward −1. Used inside blocks to add curvature.
- **Sigmoid** squeezes any number into the range 0 to 1. Think of a dimmer switch: 0 is off, 1 is fully on. Used for gates.
- **Softmax** turns a list of numbers into shares that add up to 1. Bigger numbers get bigger shares. Used when the network must "choose how much attention to give each item".

### 4.2 Residual (skip) connection

A residual connection adds the input back to the output of a block: `output = input + block(input)`. Intuition: the block only has to learn a correction, not the whole answer, and information has a shortcut path so it does not get lost in deep networks.

### 4.3 LayerNorm

**LayerNorm** (`nn.LayerNorm`) rescales the numbers inside one row to have mean 0 and standard deviation 1, then applies a small learned stretch and shift. Intuition: it keeps the internal numbers in a comfortable range so training stays stable.

### 4.4 GLU: the gate (a valve)

A **gated linear unit** splits a signal into two halves, a and b, and outputs `a × sigmoid(b)`. The sigmoid half is a valve between 0 and 1 that decides how much of the signal a goes through. In code (`modeling/nets.py`):

```python
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)      # produce twice as many numbers: a and b

    def forward(self, x):
        return F.glu(self.fc(x), -1)       # split in half, return a * sigmoid(b)
```

The reason for a gate is that the network can learn to close the valve for parts that are not useful, effectively skipping them.

### 4.5 GRN: the gated residual network (the TFT workhorse)

The **GRN** is a small package used many times in TFT and in TGT-M. It does: make a transformation, apply a gate to it, add it back to the input (residual), and normalise. Optionally it can take an extra **context** vector `c` to flavour the transformation. The code:

```python
class GRN(nn.Module):
    """LayerNorm(x + GLU(W2 ELU(W1 x + Wc c)))"""
    def __init__(self, d, d_ctx=None, drop=0.1):
        super().__init__()
        self.w1 = nn.Linear(d, d)
        self.wc = nn.Linear(d_ctx, d, bias=False) if d_ctx else None
        self.w2 = nn.Linear(d, d)
        self.glu = GLU(d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, c=None):
        h = self.w1(x)
        if self.wc is not None and c is not None:
            h = h + self.wc(c)                 # add the context, if there is one
        h = self.drop(self.w2(F.elu(h)))       # bend, transform, randomly switch off a share
        return self.norm(x + self.glu(h))      # gate it, add the input back, normalise
```

Read the last line as the recipe: "keep the original `x`, add a gated improvement, and tidy the scale".

### 4.6 GRU and GRU-D: a diary reader that understands missing pages

A **GRU** (gated recurrent unit) reads a sequence one step at a time and keeps a **memory** (the hidden state `h`). At every step it decides, with gates, how much of the old memory to keep and how much of the new step to write in. After the last step the memory summarises the sequence. In PyTorch one step of a GRU is `nn.GRUCell`.

A plain GRU does not know about missing values. **GRU-D** (the D is for decay) adds two ideas for gaps:

1. **A mask** says whether the value at this step was really observed.
2. **Time since the last observation**: the longer ago the last real reading, the more the model should distrust the old memory and the old value. So the old memory and the old value are *decayed* (shrunk) by an amount that grows with the gap.

The decay is learned: `exp(−relu(w · gap + b))` is a number between 0 and 1 that gets smaller as the gap grows. In our code the file is `modeling/nets.py`, class `GRUD`:

```python
class GRUD(nn.Module):
    def __init__(self, n_ch, d):
        super().__init__()
        self.d = d
        self.gx = nn.Linear(1, 1)              # learns how fast the last VALUE should decay
        self.gh = nn.Linear(1, d)              # learns how fast each memory cell should decay
        self.cell = nn.GRUCell(n_ch, d)        # the ordinary GRU step (10 inputs -> d memory cells)

    def forward(self, seq):
        B, T, _ = seq.shape                    # B rows, T = 13 steps
        h = seq.new_zeros(B, self.d)           # memory starts empty
        x_last = seq.new_zeros(B, 1)           # the last real reading seen so far (relative: 0 = baseline)
        delta = seq.new_zeros(B, 1)            # steps since the last real reading
        prev_m = seq.new_ones(B, 1)            # was the previous step observed?
        out = []
        for t in range(T):
            x, m = seq[:, t, 0:1], seq[:, t, 1:2]       # the reading (channel 0) and its mask (channel 1)
            delta = torch.where(prev_m > 0, torch.ones_like(delta), delta + 1) if t else delta
            dn = delta / 12.0                                   # scale the gap
            gamma_x = torch.exp(-F.relu(self.gx(dn)))           # value decay in (0, 1]
            x_hat = m * x + (1 - m) * gamma_x * x_last          # real value if seen, else decayed last value
            gamma_h = torch.exp(-F.relu(self.gh(dn)))           # memory decay in (0, 1]
            h = self.cell(torch.cat([x_hat, m, seq[:, t, 2:]], -1), h * gamma_h)
            x_last = torch.where(m > 0, x, x_last)              # remember the newest real reading
            prev_m = m
            out.append(h)
        return torch.stack(out, 1)             # one memory state per step: (B, 13, d)
```

Walk through one loop iteration in words. At step t the code takes the reading `x` and the mask `m`. If the previous step was observed, the gap `delta` is reset to 1; otherwise it grows by one. The gap is scaled and fed through two small linear layers to produce two decay factors, each between 0 and 1. If the reading is observed (`m = 1`), `x_hat` is the reading. If it is missing (`m = 0`), `x_hat` is the *last real reading shrunk by the decay*, which fades toward 0, and 0 means "exactly at baseline" because our readings are relative to the baseline. The memory `h` is also shrunk by its decay before the GRU cell updates it with the new input: `[x_hat, mask, the 8 other channels]`, which is 1 + 1 + 8 = 10 numbers. We save the memory at every step in `out`. The result has one hidden state per step, shape (batch, 13, d).

Important: the GRU-D is the **encoder** of TGT-M. It turns the raw diary into 13 learned summaries, one per step, that already account for the gaps.

### 4.7 Variable selection: which facts matter?

The static vector has 88 numbers. The **variable-selection network** (VSN, from TFT) learns how much to listen to each of them and combines them into one summary vector of size `d`, the **context** `c`. It does this in three moves:

1. Give every one of the 88 numbers its own little embedding (a vector of size `d`): `embedding_v = ELU(x_v × W_v + b_v)`. Each variable has its own `W_v` and `b_v`, so each one is transformed in its own way. (ELU is the per-variable bend that distinguishes our final design from the first one.)
2. Compute 88 **weights** that add up to 1, by looking at the whole vector: a small network `sel` (linear, ELU, linear) followed by softmax. These weights are the "how much do I listen to this variable" numbers.
3. Take the weighted sum of the 88 embeddings and pass it through a GRN. The result is the context `c`.

Code (`nets.py`, classes `VSN` and `VSN2`):

```python
class VSN(nn.Module):
    def __init__(self, n_vars, d, drop=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_vars, d) * 0.1)      # one weight vector per variable
        self.b = nn.Parameter(torch.zeros(n_vars, d))            # one bias vector per variable
        self.sel = nn.Sequential(nn.Linear(n_vars, d), nn.ELU(), nn.Linear(d, n_vars))
        self.grn = GRN(d, drop=drop)

    def forward(self, x):                                        # x: (B, 88)
        emb = x.unsqueeze(-1) * self.W + self.b                  # (B, 88, d)
        w = torch.softmax(self.sel(x), -1)                       # (B, 88), each row sums to 1
        return self.grn((w.unsqueeze(-1) * emb).sum(1)), w       # context (B, d) and the weights

class VSN2(VSN):                                                 # the final design
    def forward(self, x):
        emb = F.elu(x.unsqueeze(-1) * self.W + self.b)           # same, but with the ELU bend per variable
        w = torch.softmax(self.sel(x), -1)
        return self.grn((w.unsqueeze(-1) * emb).sum(1)), w
```

The weights `w` are also returned. We store them and average over the test rows to see which inputs the model listens to. This is the "Top 12 inputs by variable-selection weight" chart. A caution we state honestly: the weights are close to uniform (the 24 rainfall inputs receive 26.6 percent of the weight, and a perfectly uniform split would give 27.3 percent), so these weights are a diagnostic and **not** proof that rainfall dominates.

### 4.8 Static enrichment

The diary summaries (13 states of memory) are now **enriched** with the context `c` from the fact sheet. This is a GRN that takes each of the 13 memory states as `x` and the context `c` (copied 13 times) as the extra input. Intuition: "read the diary while keeping the well's facts in mind". In code: `E = self.enrich(H, c.unsqueeze(1).expand_as(H))`.

### 4.9 Attention: which past rounds matter?

**Attention** answers: "to predict the current round, which of the 13 steps should I look at most?" It has three roles:

- a **query**: what I am asking (here: the enriched state of the current round, step t);
- **keys**: a label on each step saying what it contains (all 13 enriched states);
- **values**: what each step contributes if chosen (also the 13 enriched states).

The query is compared with each key, the comparisons are turned into weights with softmax (they add up to 1), and the output is the weighted mix of the values. A step that matches the query gets a big weight. With **2 heads**, the model does this twice in different ways (each head works with half of the 48 numbers, 24 each) and joins the results, so it can look for two different patterns.

Two details in our code: the padding steps (before the table starts, channel 6 = 0) are blocked with `key_padding_mask` so they receive no attention; and we keep the attention weights (`self.last_attn`), which show how much the current round looked at each of the 13 steps. This is the "Mean attention weight by step" chart.

Our attention uses only the current round as the query (a single query), which keeps it simple and makes its weights easy to read.

### 4.10 Quantile outputs and the loss functions

TGT-M produces four numbers per row: a **point estimate** of z, and the **10%, 50% and 90% quantiles**. A quantile answers "what value is z below with this probability". If the 10% quantile is −0.8 and the 90% quantile is +1.2, the model is saying "I am 80 percent sure that z is between −0.8 and 1.2". That band is the **80% prediction interval**, which is a real strength of this model because it says how uncertain the estimate is.

To teach the point estimate we use the **Huber loss**: for small errors it behaves like squared error, for large errors like absolute error, so outliers do not dominate. To teach the quantiles we use the **pinball loss**: for a quantile level τ and error e = (true − predicted), the loss is `max(τ·e, (τ − 1)·e)`. For τ = 0.9 it punishes under-prediction 9 times more than over-prediction, which pushes the prediction up to the level that 90 percent of the true values fall below. The code (`nets.py`):

```python
QUANTILES = (0.1, 0.5, 0.9)

def quantile_loss(q, y):
    loss = 0
    for i, tau in enumerate(QUANTILES):
        e = y - q[:, i]
        loss = loss + torch.maximum(tau * e, (tau - 1) * e).mean()
    return loss / len(QUANTILES)
```

The total training loss is `Huber(point) + 0.5 × pinball`.

---

## Part 5. TGT-M step by step

Now we assemble the pieces. The model is the class `TGTM2` in `modeling/nets.py` (it is registered under the name `"TGT-M"`; the first design, class `TGTM`, is kept as `"TGT-M-v1"`). The picture in the report and slides shows the same flow.

### 5.1 The whole flow in one paragraph

The diary (13 × 10) goes into **GRU-D**, which produces 13 memory states. The fact sheet (88 numbers) goes into the **variable-selection network**, which produces one context vector. The context is mixed into each of the 13 memory states by **static enrichment**. The enriched states go to **attention**, where the current round asks which past rounds matter. The attention result is added to the current round's state through a **gate** and normalised. The **context skip** then adds the well facts once more. A final **GRN** polishes the result, and a **linear output layer** produces four numbers: point, q10, q50, q90. Those become the depth and the interval.

### 5.2 The exact code of the model

```python
class TGTM(nn.Module):                                   # first design; TGTM2 inherits from it
    def __init__(self, n_static, hidden=48, drop=0.1, n_ch=10, heads=2, rain_idx=None, **_):
        super().__init__()
        d = hidden
        self.enc = GRUD(n_ch, d)                         # 1. diary reader
        self.vsn = VSN(n_static, d, drop)                # 2. fact-sheet selector
        self.enrich = GRN(d, d_ctx=d, drop=drop)         # 3. mix facts into every step
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)   # 4. which rounds matter
        self.gate = GLU(d)                               # 5. valve on the attention result
        self.norm = nn.LayerNorm(d)
        self.pos = GRN(d, drop=drop)                     # 7. final polish
        self.out = nn.Linear(d, 1 + len(QUANTILES))      # 8. four outputs: point, q10, q50, q90
        ...

class TGTM2(TGTM):
    """TGT-M v2: per-variable VSN non-linearity and a skip of the selected static context to the output."""
    def __init__(self, n_static, hidden=48, drop=0.1, n_ch=10, heads=2, rain_idx=None, **_):
        super().__init__(n_static, hidden, drop, n_ch, heads, rain_idx)
        self.vsn = VSN2(n_static, hidden, drop)          # the ELU-per-variable selector
        self.mix = nn.Linear(2 * hidden, hidden)         # 6. context skip

    def forward(self, seq, st, return_all=False):        # seq: (B,13,10)   st: (B,88)
        H = self.enc(seq)                                # (B,13,d)  one memory per step
        c, w = self.vsn(st)                              # c: (B,d)  w: (B,88)
        self.last_w = w.detach()                         # keep the variable weights to look at later
        E = self.enrich(H, c.unsqueeze(1).expand_as(H))  # (B,13,d)  facts mixed into every step
        pad = seq[:, :, 6] < 0.5                         # steps that do not exist (padding)
        a, aw = self.attn(E[:, -1:], E, E, key_padding_mask=pad, need_weights=True)   # query = current step
        z = self.norm(E[:, -1] + self.gate(a.squeeze(1)))                             # (B,d)
        z = self.norm(z + F.elu(self.mix(torch.cat([z, c], -1))))                     # context skip
        o = self.out(self.pos(z))                        # (B,4)
        self.last_attn = aw.detach()                     # keep the attention weights to look at later
        return (o[:, 0], o[:, 1:]) if return_all else o[:, 0]
```

### 5.3 Stage by stage, with shapes and parameters

Take a batch of B rows. The table gives the shape of the data after each stage (for Model B, d = 48) and the number of parameters in that block.

| # | stage | code | output shape | parameters |
|---|---|---|---|---|
| 0 | inputs | `seq`, `st` | (B,13,10) and (B,88) | 0 |
| 1 | GRU-D encoder | `self.enc(seq)` | (B,13,48) | 8,738 |
| 2 | variable selection | `self.vsn(st)` | context (B,48), weights (B,88) | 26,536 |
| 3 | static enrichment | `self.enrich(H, c...)` | (B,13,48) | 11,808 |
| 4 | attention, 2 heads | `self.attn(E[:, -1:], E, E, ...)` | (B,1,48) and weights (B,1,13) | 9,408 |
| 5 | gate + residual + LayerNorm | `self.norm(E[:, -1] + self.gate(...))` | (B,48) | 4,704 (gate) |
| 6 | context skip | `self.mix(cat[z, c])` | (B,48) | 4,656 |
| 7 | position-wise GRN | `self.pos(z)` | (B,48) | 9,504 |
| 8 | output heads | `self.out(...)` | (B,4) | 196 |
| | the shared LayerNorm | `self.norm` (used twice) | | 96 |
| | **total** | | | **75,646** |

The biggest block is the variable-selection network (26,536) because it has to handle 88 variables, each with its own weight and bias vectors (88 × 48 × 2 = 8,448 of them). The GRU-D has only 8,738 parameters, of which the GRU cell is 8,640 and the two decay layers are 98.

Why does the sequence step count stay 13 until attention? Because GRU-D produces one state per step. Attention then collapses the 13 into one by taking a weighted mix, and from there on we work with one vector per row.

### 5.4 Walking one row through the model

Imagine the Cherfully well, round May 2006.

1. Its diary has 13 rows: the 12 rounds before May 2006 and May 2006 itself (hidden reading). Most past rounds have a reading, one or two may be missing, which shows as mask 0.
2. GRU-D reads from the oldest round to the newest. For a missing round it fades the last real reading toward baseline and shrinks the memory. After 13 steps it has produced 13 memory vectors of 48 numbers.
3. At the same time the 88 well facts go through variable selection and become one 48-number context.
4. The context is mixed into all 13 memory vectors.
5. The current round (May 2006) looks at the 13 vectors with attention. Suppose it gives a lot of weight to the previous May and to last round; those two contribute most.
6. The attended result is added to May 2006's own vector through the gate, the well's facts are added again, a last GRN polishes it, and the output layer gives four numbers, for example point z = 1.8, q10 = 0.4, q50 = 1.7, q90 = 3.0.
7. Depth = baseline mean + z × SD = 1.684 + 1.8 × 0.376 = 2.36 m, and the class from z = 1.8 is Moderate Stress. (These four numbers are only an illustration; the real model's output for this row is in `results/`.)

### 5.5 Why this design? (what each piece is for)

- **GRU-D** exists because 13.6 percent of the readings are missing and the gaps are informative; a model should not pretend they are real values.
- **Variable selection** exists because there are 88 inputs, many of them similar, and we want the model to be able to ignore the useless ones and to tell us which ones it used.
- **GRN and gates** exist because they let the network suppress what it does not need, which helps on a small tabular problem.
- **Static enrichment** lets the sequence be read in the light of the well's permanent facts.
- **Attention** lets the model pick the rounds that matter (for example the same season last year) instead of squeezing everything through the last memory state, and it shows us what it picked.
- **Context skip** (added in version 2) gives the well facts a direct road to the output.
- **Quantile outputs** give an honest uncertainty band.

An honest note on naming: TGT-M uses the **building blocks** of the Temporal Fusion Transformer (variable selection, GRN, static enrichment, gated interpretable attention, quantile outputs) but it is not the full TFT: it replaces the TFT's LSTM encoder by the GRU-D encoder, predicts one value (the hidden current round) instead of several future steps, and uses a single query. That is why we call it a fusion of GRU-D and TFT parts.

---

## Part 6. Training: the code in `modeling/lib.py`

The function is `fit_deep(kind, bundle, y, tr, va, cfg, seed, epochs, patience)`. Here is the skeleton with explanations.

```python
def fit_deep(kind, bundle, y, tr, va, cfg, seed, epochs=30, patience=4, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)                          # make the run repeatable
    net = N.NETS[kind](n_static=bundle.X.shape[1], rain_idx=..., **cfg["arch"])   # build the model
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    huber = nn.HuberLoss(delta=1.0)
    bs = 1024                                                              # batch size
    for ep in range(epochs):
        net.train()                                                        # dropout on
        perm = np.random.permutation(train_row_indices)                    # shuffle the training rows
        for i in range(0, len(perm), bs):                                  # one batch at a time
            j = perm[i:i + bs]
            if kind.startswith("TGT-M"):
                pt, q = net(St[j], Xt[j], return_all=True)                 # point and quantiles
                loss = huber(pt, yt[j]) + 0.5 * N.quantile_loss(q, yt[j])  # the TGT-M loss
            else:
                loss = huber(net(St[j], Xt[j]), yt[j])                     # other nets: point only
            opt.zero_grad(); loss.backward()                               # compute the gradients
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)                # clip huge nudges
            opt.step()                                                     # nudge the weights
        net.eval()                                                         # dropout off
        vl = mean absolute error on the validation rows
        if vl < best - 1e-4: keep a copy of the weights, reset the counter
        else: counter += 1; stop if counter >= patience
    net.load_state_dict(best_state)                                        # go back to the best epoch
    return net, best, epochs_run
```

The target `y` is the capped anomaly z. The model never sees the true z of the current round in its input, only as the answer to learn from.

### 6.1 Settings and how they were chosen

For each model we tried 3 to 4 settings (called a **grid**), and chose the one with the lowest error on the **validation** rows, never on the test rows. The grid for TGT-M (`GRIDS["TGT-M"]` in `lib.py`) has three settings of hidden size, dropout, learning rate and weight decay:

| setting | hidden | dropout | learning rate | weight decay |
|---|---|---|---|---|
| 0 | 32 | 0.2 | 0.001 | 0.001 |
| 1 | 32 | 0.3 | 0.0005 | 0.001 |
| 2 | 48 | 0.3 | 0.0005 | 0.01 |

The chosen settings were: Model A setting 0 and Model B setting 2 (so 75,646 parameters for Model B). Every setting is trained for up to 20 epochs with patience 3, with 3 seeds, and the reported numbers are averages over the seeds. The script that trains one TGT-M job is `modeling/tgtm_job.py`.

### 6.2 The design iteration, stated openly

The first TGT-M design (version 1) overfitted quickly (validation error 0.841 for Model A). We changed the architecture and the regularisation (the ELU per variable in the variable-selection network, the context skip, dropout and weight decay) and the validation error improved to 0.819 for Model A. We chose between the two **on validation only** and we say so in the report, because iterating on the design is allowed as long as the test set is not used to decide.

### 6.3 After training: `predict_deep`

`predict_deep` runs the trained network on rows without changing anything (`torch.no_grad()`) and returns the predicted z, and for TGT-M also the quantiles, the variable-selection weights and the attention weights. These are saved and used by the evaluation scripts and by the notebook charts.

---

## Part 7. From the model's output to the answers

1. **Depth.** `depth_from_z(meta, z)` in `lib.py` returns `max(baseline_mean + z × sd, 0)` (a depth cannot be negative).
2. **Stress class.** The thresholds 1, 1.5 and 2 on z (`stress_label` in `data.py`). Because regression predictions shrink toward zero, we also report a version with class thresholds **calibrated on the validation rows only**.
3. **Interval.** The 10% and 90% outputs form the 80% interval. We measure how often the true z falls inside (coverage, 79.2 percent, close to the 80 percent promised) and how narrow the band is (width, 2.32 z-units).
4. **Metrics.** `modeling/metrics_full.py` computes RMSE, MSE, MAE, MAPE, R², NRMSE, MedAE and bias for the depth, and accuracy, precision, recall, F1, specificity, AUC, MCC, kappa and balanced accuracy for the four classes. The formulas are in Section 3.5 of the report.
5. **Significance.** The difference between two models is tested with a paired bootstrap over wells (2,000 resamples), because the many rows of one well are not independent.

---

## Part 8. The other models, so you can say how TGT-M differs

All models get the same rows, the same target, the same splits, and each is tried with and without rainfall.

- **ANN** (`MLP`): a plain multilayer perceptron that looks only at the 88-number fact sheet. No diary.
- **LSTM** (`LSTMNet`): a recurrent network that reads the diary (13 × 10) plus the fact sheet. Unlike GRU-D it has no special treatment of gaps, no variable selection, no attention.
- **1D-CNN** (`CNN1D`): slides small filters along the diary to detect local patterns, then combines with the fact sheet.
- **Random Forest, XGBoost, SVR**: tree ensembles and a kernel machine on the fact sheet (SVR uses a Nystroem approximation so it runs on 100,000 rows). They have no diary.
- **Naive baselines**: predict z = 0 (the seasonal baseline), repeat the previous round, or repeat the same season last year. A good model must beat these.

What TGT-M adds over the LSTM: the explicit handling of gaps (mask and decay), a learned selection over the inputs, attention over the past, and quantile outputs.

---

## Part 9. What it achieved, honestly

On the test split (2018 to 2022, 40,630 rows), the depth error (MAE) in Model B is: LSTM 1.104 m, 1D-CNN 1.113, XGBoost 1.121, ANN 1.125, **TGT-M 1.129**, Random Forest 1.150, SVR 1.180. TGT-M is **fifth of seven**. We do not claim it is the most accurate.

What is supported by pre-registered tests (written before the tests were run):

- **H1 (intervals): supported.** TGT-M's 80% intervals have a better interval score than quantile XGBoost and than the conformal intervals of all six other models.
- **H2a (missing history): supported.** When 0 to 60 percent of the recent readings are removed at test time, TGT-M loses less accuracy than the LSTM and the 1D-CNN.
- **H2b: not supported.** The model with the lowest error at 60 percent removal is the Random Forest, not TGT-M.

Other findings: rainfall lowers the error of every model (8 to 19 percent); the error on unseen states is within about 2 percent of the error on seen states, and TGT-M's change is the smallest (a post-hoc observation). Limits: the anomaly is only weakly predictable (R² about 0.3), so four-class F1 stays low for every model; held-out wells keep their own earlier readings, so cold-start wells are not tested; the hyperparameter search is small.

---

## Part 10. Where everything is, and questions you may be asked

### 10.1 File map

| what | where |
|---|---|
| how the diary and the fact sheet are built | `modeling/data.py`, function `build`, class `Scaler` |
| the model blocks and TGT-M | `modeling/nets.py`: `GLU`, `GRN`, `VSN`, `VSN2`, `GRUD`, `TGTM`, `TGTM2`, `quantile_loss` |
| scaling, training, prediction, settings | `modeling/lib.py`: `Bundle`, `fit_deep`, `predict_deep`, `GRIDS`, `depth_from_z`, `evaluate` |
| one TGT-M training job | `modeling/tgtm_job.py` |
| full metric tables | `modeling/metrics_full.py` |
| the Spark check | `modeling/spark_check.py` |
| the live demonstration | `notebooks/3_Review2_Implementation.ipynb` (sections 2 and 5 build and train the model) |

### 10.2 A tiny experiment you can run yourself

Open the notebook, section 2, and look at the table of output shapes: it is the table in Part 5.3 produced by the real model. In section 5 the mini training shows the validation error going from 0.96 to 0.84 in 8 epochs, below the 0.965 of always predicting zero.

### 10.3 Likely questions and short answers

1. **What does TGT-M stand for?** Temporal Gated Transformer Model. It is a fusion of GRU-D and parts of the Temporal Fusion Transformer.
2. **What is the input?** A 13-step diary with 10 channels and a fact sheet of 88 numbers (55 without rainfall). The current reading is hidden.
3. **What is the output?** The anomaly z of the hidden reading (a point value) and its 10%, 50% and 90% quantiles. Depth and class follow from z.
4. **Why predict z instead of depth?** It is comparable across wells because it is relative to each well's own past-only baseline.
5. **How do you avoid leakage?** The baseline, lags and rainfall use only earlier data; rainfall ends at month M−1; scalers are fitted on training rows; splits are by time and by state; there are tests that change the future and check the inputs do not move.
6. **Why GRU-D?** Because 13.6 percent of readings are missing and GRU-D models the gap with a mask and a decay instead of inventing values.
7. **What does the variable-selection network do?** It gives each of the 88 inputs its own embedding and learns softmax weights that say how much to listen to each; the weights are a diagnostic.
8. **What does attention do here?** The current round asks which of the 13 steps matter; the answer is a set of 13 weights that add up to 1, which we plot.
9. **What is a GRN and why a gate?** A gated residual network: transform, gate, add back the input, normalise. The gate lets the network close off what it does not need.
10. **What are the losses?** Huber loss for the point estimate and pinball loss for the quantiles, total = Huber + 0.5 × pinball.
11. **How many parameters?** 75,646 for the Model B setting (hidden size 48).
12. **How did you tune it?** Three settings, chosen on the validation years 2016 and 2017 only; three seeds; the test years were used once.
13. **Is TGT-M the best model?** No. It is fifth of seven on depth error; its supported strengths are the 80% intervals and robustness to missing recent readings.
14. **What does rainfall do?** It lowers the error of every model by 8 to 19 percent, which is why we train each model with and without it (Model A and Model B).
15. **What is the limit of the model?** The anomaly is only weakly predictable (R² about 0.3), the four-class F1 is low for all models, cold-start wells were not tested, and the search is small.

### 10.4 Glossary

- **anomaly z**: how many SDs a reading is above the well's own past-only baseline for that season.
- **attention**: a way for one item to look at all others and weigh them.
- **baseline**: mean and SD of the earlier readings of the same well and season.
- **batch**: a group of rows processed together (1,024).
- **channel**: one of the 10 numbers per step of the diary.
- **context**: the summary vector of the fact sheet produced by variable selection.
- **dropout**: randomly switching off a share of numbers during training to avoid overfitting.
- **epoch**: one pass over all training rows.
- **gate**: a sigmoid valve between 0 and 1.
- **GRN**: gated residual network.
- **GRU-D**: gated recurrent unit with decay and a mask for missing values.
- **leakage**: letting the model see information from the future or from the answer.
- **mask**: 1 if the reading was observed, 0 if it is missing or hidden.
- **parameter**: a weight or bias learned in training.
- **quantile**: the value below which a given share of outcomes falls.
- **residual connection**: adding the input back to a block's output.
- **round**: one measurement occasion (January, May, August or November).
- **seed**: the starting point of random numbers; we train with three.
- **VSN**: variable-selection network.
