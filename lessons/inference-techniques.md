# LLM Inference Techniques Index

This is the cross-lesson index for concrete inference techniques. Each link points to the section that explains the mechanism, not merely a passing mention.

## Model partitioning

| Technique                         | What it is used for                                                                 | Lesson                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Column-parallel linear layer      | Split output features; optionally all-gather output shards                          | [[llm-inference-parallelism#Question 1 — Tensor parallelism\|Tensor-parallel question 1]]                   |
| Row-parallel linear layer         | Split the contraction dimension; sum partial outputs                                | [[llm-inference-parallelism#Question 2 — Row parallelism\|Tensor-parallel question 2]]                      |
| Column-to-row MLP composition     | Keep the expanded MLP activation sharded and use one final all-reduce               | [[llm-inference-parallelism#Question 3 — Composing tensor-parallel layers\|Composing TP layers]]            |
| Attention-head tensor parallelism | Keep complete attention heads local until the output projection                     | [[llm-inference-parallelism#Lesson 1 — Tensor parallelism for attention\|Tensor parallelism for attention]] |
| GQA/MQA KV-head replication       | Continue query-head TP beyond the number of distinct KV heads                       | [[llm-inference-parallelism#Check 1 — Grouped-query attention\|Grouped-query attention]]                    |
| Pipeline parallelism              | Split model layers into stages and send activations between stages                  | [[llm-inference-parallelism#Question 5 — Hybrid parallelism and topology\|Hybrid parallelism and topology]] |
| Topology-aware TP/PP placement    | Keep frequent TP collectives on fast links and stage boundaries across slower links | [[llm-inference-parallelism#Question 5 — Hybrid parallelism and topology\|Hybrid parallelism and topology]] |

## Numerical compression

| Technique | What it is used for | Lesson |
|---|---|---|
| Group-wise integer weight quantization | Store integer weight codes and reconstruct them using a shared group scale; trade grid resolution against range | [[llm-inference-data-types#1.1 Integer quantization: equally spaced levels within a group\|Scaled INT4 and INT8 weight encodings]] |
| NF4 weight quantization | Store 4-bit indices into a fixed nonuniform codebook, with a separate block scale | [[llm-inference-data-types#1.3 NF4: choose the levels for a distribution\|NF4 encoding and its trade-off]] |
| Block-scaled FP4 quantization | Share scales across E2M1 elements; MXFP4 and NVFP4 use different block sizes and scale encodings | [[llm-inference-data-types#3.2 MXFP4 versus NVFP4: same element levels, different scaling\|MXFP4 and NVFP4 scaling]] |
| Fused weight-only dequantization | Decode small compressed weight tiles inside the compute path instead of keeping a full wider-precision weight copy | [[llm-inference-data-types#4.2 Follow an illustrative W4A16 operation\|Illustrative W4A16 kernel path]] |
| Higher-precision accumulation | Use wider partial sums than the multiplication operands or stored output to reduce additional rounding | [[llm-inference-data-types#4.2 Follow an illustrative W4A16 operation\|Operand, accumulator, and output precision]] |
| FP8 KV-cache quantization | Compress persistent attention state separately from weights; attention compute precision depends on the backend | [[llm-inference-data-types#4.3 KV-cache precision is another decision\|Cache storage versus attention arithmetic]] |

## Decoding

| Technique | What it is used for | Lesson |
|---|---|---|
| Speculative decoding | Evaluate a proposed continuation at several positions in one target-model forward pass | [[speculative-decoding#1. How one forward pass checks several proposed tokens\|Draft proposals and target verification]] |
| Speculative KV-cache rollback | Restore the retained prefix, valid lengths, and block mappings after rejecting proposals | [[speculative-decoding#3.1 How the target cache changes\|Target-cache validity]]; [[speculative-decoding#3.3 Rollback in a paged cache\|Paged rollback and reclamation]] |
| Draft-cache catch-up | Process committed tokens missing from the draft cache before it predicts the next proposal | [[speculative-decoding#3.2 Why the draft cache can be behind\|Separate target and draft cache progress]] |
| Standalone-model drafting | Generate speculative proposals with a smaller autoregressive LM, optionally distilled for the target | [[speculative-decoding#5.1 A standalone smaller language model\|Independent smaller draft models]] |
| EAGLE-style drafting | Reuse target features in a trained lightweight auxiliary drafter | [[speculative-decoding#5.2 A target-conditioned auxiliary drafter: EAGLE\|Target-conditioned drafters]] |
| Medusa heads | Predict multiple future positions from one target state using lightweight parallel heads | [[speculative-decoding#5.3 Multiple future-token heads: Medusa\|Medusa proposal mechanism]] |
| Native multi-token prediction | Use prediction modules trained with a model as speculative proposal machinery | [[speculative-decoding#5.4 Native multi-token prediction: DeepSeek-style MTP\|DeepSeek-style MTP and its conditioning]] |
| Prompt lookup / n-gram speculation | Retrieve a continuation following an earlier matching token suffix for target verification | [[speculative-decoding#5.5 Prompt lookup and n-gram proposals\|Model-free proposals from existing text]] |

## Performance and communication analysis

| Technique | What it is used for | Lesson |
|---|---|---|
| Decode roofline analysis | Separate compute and HBM bounds using arithmetic intensity | [[llm-inference-parallelism#Lesson 2 — Decode-step roofline\|Decode-step roofline]] |
| Speculative-decoding output accounting | Distinguish committed output, accepted drafts, proposed drafts, and verification query positions | [[speculative-decoding#4.1 Count the output each round actually produces\|Useful output per speculative round]] |
| Speculative-decoding break-even analysis | Compare complete round time with ordinary decoding of the same useful output | [[speculative-decoding#4.2 Compare the whole round with ordinary decoding\|Round cost and speedup]] |
| Accepted-prefix survival analysis | Estimate output per round from whole-prefix acceptance, with explicit probability assumptions | [[speculative-decoding#4.3 Derive the average number of tokens per round\|Expected output and proposal length]] |
| Critical-batch derivation | Find when a weight-streaming decode matmul becomes compute-bound | [[llm-inference-parallelism#Derivation — critical batch size\|Critical batch size]] |
| Ring all-reduce | Sum distributed partial outputs with reduce-scatter plus all-gather | [[llm-inference-parallelism#Lesson 3 — Ring all-reduce cost and TP scaling\|Ring all-reduce cost]] |
| TP scaling cutoff | Compare saved local work against added collective startup and transfer time | [[llm-inference-parallelism#Combining local work and communication\|Combining local work and communication]] |

## KV-cache distribution and distributed attention

| Technique | What it is used for | Lesson |
|---|---|---|
| Batch sharding | Place complete caches for different requests on different ranks | [[kv-cache-sharding#2. Which axis are we sharding?\|KV sharding axes]] |
| KV-head sharding | Place complete histories for different KV heads on different ranks | [[kv-cache-sharding#2. Which axis are we sharding?\|KV sharding axes]] |
| Context/position sharding | Split one head's token positions across ranks | [[kv-cache-sharding#3. Context parallelism is not Ring Attention\|Context parallelism]] |
| KV replication | Duplicate KV state to simplify local attention or satisfy a TP layout | [[kv-cache-sharding#2. Which axis are we sharding?\|KV sharding axes]] |
| Pass-query decode | Send a small decode query to stationary KV shards and merge partial states | [[kv-cache-sharding#4. Pass-query decode: derive the data movement\|Pass-query decode]] |
| Online-softmax state merge | Combine position-sharded attention exactly with $(m,\ell,n)$ states | [[kv-cache-sharding#5. Exact attention over position shards\|Exact attention over position shards]] |
| Two-phase standard-collective merge | Max-reduce, rescale locally, then sum-reduce denominator and numerator | [[kv-cache-sharding#7. Mapping pass-query decode onto collectives\|Mapping pass-query onto collectives]] |
| K/V all-gather | Materialize complete K/V where distinct local query work justifies it | [[kv-cache-sharding#11.4 K/V all-gather payload\|K/V all-gather payload]] |
| Ring/pass-KV attention | Stream K/V blocks through query owners without full materialization | [[kv-cache-sharding#11.5 Ring/pass-KV payload\|Ring/pass-KV payload]] |
| Sequence-to-head all-to-all | Convert sequence shards into complete head assignments | [[kv-cache-sharding#8. Compare context-parallel schedules\|Context-parallel schedule comparison]] |
| Hierarchical softmax-state reduction | Merge inside fast topology groups before crossing slower links | [[kv-cache-sharding#13. Topology-aware design application\|Topology-aware design]] |

## Production serving

| Technique                         | What it is used for                                                            | Lesson                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Continuous batching               | Rebuild batch membership at model-iteration boundaries                         | [[production-llm-serving#Node 3 — Eligibility comes before policy\|Eligibility and continuous batching]]                      |
| Multi-budget iteration scheduling | Limit sequences, input tokens, estimated work, and KV growth separately        | [[production-llm-serving#Node 4 — One iteration needs several budgets\|Iteration budgets]]                                    |
| Paged KV allocation               | Map logical KV blocks to noncontiguous physical blocks                         | [[production-llm-serving#Node 6 — KV capacity comes from a finite block pool\|KV block pool]]                                 |
| KV-aware schedule reservation     | Reserve writable blocks before launching selected model work                   | [[production-llm-serving#Node 5 — Selection must reserve writable KV capacity\|Selection and allocation]]                     |
| Capacity-aware admission          | Choose conservative, incremental, or predicted reservation for future growth   | [[production-llm-serving#Node 7 — Current fit is not a completion guarantee\|Admission and future growth]]                    |
| Chunked prefill                   | Split long prompts across iterations to create decode scheduling opportunities | [[production-llm-serving#Node 8 — Continuous batching does not bound iteration time\|Prefill/decode interference]]            |
| Prefix caching                    | Share immutable computed KV for equivalent prefixes                            | [[production-llm-serving#Node 9 — A cached KV block can be reclaimable without being empty\|Prefix reuse]]                    |
| Copy-on-write prefix sharing      | Preserve isolation when shared state would otherwise be modified               | [[production-llm-serving#Shared-prefix example\|Shared-prefix example]]                                                       |
| Active-KV preemption with prefill | Free active KV, then run prefill over committed token history to rebuild it    | [[production-llm-serving#Node 10 — Evicting cached state and preempting active state are different\|Active-state preemption]] |
| Active-KV preemption by offload   | Move active KV to another memory tier and reload it later                      | [[production-llm-serving#Node 10 — Evicting cached state and preempting active state are different\|Active-state preemption]] |
| Deferred KV reclamation            | Stop cancelled work immediately but reuse KV only after in-flight access finishes | [[production-llm-serving#Node 11 — Cancellation does not make in-flight KV immediately reusable\|Cancellation and deferred reclamation]] |
| Bounded streaming backpressure     | Bound per-request output buffering and feed slow-consumer state back to scheduling | [[production-llm-serving#Node 12 — Network backpressure does not automatically stop generation\|Streaming backpressure]] |
| End-to-end deadline propagation    | Preserve one absolute deadline across queues, workers, and delivery | [[production-llm-serving#Node 13 — Preserve one deadline across the whole request path\|Deadlines and overload]] |
| Resource-aware load shedding       | Reject excess work using queue, KV, and latency pressure rather than QPS alone | [[production-llm-serving#Node 13 — Preserve one deadline across the whole request path\|Deadlines and overload]] |
| Timestamp-defined latency metrics  | Define TTFT, ITL, TPOT, and E2E from explicit observation boundaries | [[production-llm-serving#Node 14 — Define metrics from timestamps, not names\|Serving metrics]] |
| Cross-layer serving telemetry      | Correlate user latency with scheduler, KV, output, and accelerator signals | [[production-llm-serving#Node 15 — User latency needs internal resource signals\|Serving diagnosis]] |
| Closed-loop and open-loop load testing | Separate fixed concurrency from independent arrivals when measuring serving capacity | [[production-llm-serving#Node 16 — The load generator can hide overload\|Load-test design]] |
| Controlled prefix-cache benchmarking | Specify cache warmth and reuse so repeated prompts do not distort comparisons | [[production-llm-serving#Node 16 — The load generator can hide overload\|Load-test controls]] |

## Worked application

[[production-llm-serving#Node 17 — Trace the batch, KV blocks, and outputs together|Integrated serving trace]] combines continuous batching, chunked prefill, block reservation, deferred cancellation cleanup, and latency accounting.

## Profiling

| Technique | What it is used for | Lesson |
|---|---|---|
| Labeled PyTorch CPU/CUDA tracing | Connect Python regions and operators to CUDA launches and GPU kernels | [[pytorch-benchmarking-profiling#Step 1 — Capture and read one inference step\|Basic capture and trace reading]] |
| Separate benchmark and profiler runs | Measure clean performance without treating capture overhead as workload latency | [[pytorch-benchmarking-profiling#Step 1 — Capture and read one inference step\|Clean timing and capture]] |
| CPU/GPU copy correlation | Distinguish a long host copy call from the actual transfer size and GPU duration | [[pytorch-benchmarking-profiling#Step 2 — Follow one Llama decode iteration\|Llama readback diagnosis]] |
| GPU stage and kernel-group breakdown | Rank execution costs by GPU time rather than CPU waits or kernel count alone | [[pytorch-benchmarking-profiling#What to analyze next\|Llama GPU cost breakdown]] |
| Perfetto SQL kernel aggregation | Group GPU kernels by name/family with explicit decode-step or graph-replay scope | [[pytorch-benchmarking-profiling#Group GPU kernels in Perfetto\|Perfetto grouping queries]] |
| Focused basic Nsight Compute report | Check resource usage of one selected hot kernel before choosing an optimization | [[pytorch-benchmarking-profiling#Step 3 — Check one GEMV before deciding to optimize it\|Logits GEMV bandwidth check]] |
| Semantic-range GPU attribution | Label a model function and sum the GPU kernels correlated with its child launches | [[pytorch-benchmarking-profiling#Step 4 — Measure GPU work attributable to RMSNorm\|RMSNorm timing]] |

## Maintenance rule

When a lesson introduces a new concrete technique:

1. add it here when its durable explanation is written;
2. link to the primary explanatory section;
3. use one canonical name and include aliases in its description;
4. do not add a technique that is only planned or mentioned in passing.
