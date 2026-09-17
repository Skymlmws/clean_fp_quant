# External storage layout

The repository uses one external-output mapping:

```text
outputs/external -> /root/autodl-tmp/clean_fp_quant/outputs
```

Project-generated artifacts live below that root:

```text
outputs/external/
├── activation-visualization/  Older and focused activation plots
├── activations/               Reusable complete BF16 activation captures
├── comparisons/               Derived metrics, plots, policies, and reports
└── logs/                      Project-level run logs
```

Shared downloaded resources are deliberately not moved into this project
directory. They may be used by other workspaces:

```text
/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers  Shared Wan model
/root/autodl-tmp/vbench                       Shared VBench evaluation models
/root/autodl-tmp/huggingface                  Shared Hugging Face cache
```

Runners may accept path overrides, but their project-output defaults must stay
under `/root/autodl-tmp/clean_fp_quant/outputs`. Do not add another repository
symlink for an individual experiment; place it under the shared output tree.
