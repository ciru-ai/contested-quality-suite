# Slim release size, measured on Linux x86-64

The archive contains the 95 selected case definitions. `./benchmark` runs the
four lightweight components (54 cases); `./benchmark --all` also runs Aider,
Hermes, and Terminal (41 cases) using their native graders and agent protocols.

| Component | Cases | Bundled component files, apparent size | Fresh Python wheel/source downloads | Installed Python environment |
| --- | ---: | ---: | ---: | ---: |
| ARC-Challenge | 18 | 9 KB selector plus a 6 KB runner | 0 | 0 |
| IFEval strict | 27 | 1.30 MB grader + 0.67 MB English tokenizer data + 11 KB selector | 4,297,561 bytes | 15,916 KiB allocated |
| HumanEval+ | 5 | 409 KB selected tasks and reference outputs plus small grader code | 775,620 bytes | 2,764 KiB allocated |
| Tool-Eval | 4 | 1.92 MB source | 3,067,149 bytes | 12,604 KiB allocated |
| Aider Polyglot, optional | 26 | 4.27 MB source and selected exercises | No default download | Slim image built on the measurement host: 4.11 GB virtual (prior native image: 7.3 GB) |
| HermesAgent, optional | 11 | 158 KB runner and verifier | No default download | Browser-free image built on the measurement host: 1.21 GB virtual (prior image: 4.52 GB) |
| Terminal-Bench, optional | 4 | 550 KB runner and selected tasks | No default download | Four native task images measured at 611 MB, 1.47 GB, 181 MB, and 619 MB on the measurement host |

The previously measured release archive was about **2.3 MB**. Its fresh extraction occupied
**11,352 KiB** on the measured filesystem. Fresh extraction plus the three
default Python environments occupied **42,660 KiB (41.7 MiB)**. Their clean wheel
and source downloads totalled **8,140,330 bytes (7.76 MiB)**.

Measurement: Python 3.12 on the current Linux x86-64 host; `pip download`
resolved each pinned/default component dependency group into separate empty
wheelhouses, and the byte totals are the downloaded files. `du -sk` measured
the installed environments and clean extraction. These numbers exclude the
external inference server, model weights, package-manager/runtime prerequisites,
and caches. Different Python versions or platforms can select different wheels.
The optional Aider and Hermes images were built from this release's Dockerfiles
on the measurement host. Together with the four previously measured Terminal task images,
their virtual sizes total about **8.20 GB** versus about **14.70 GB** for the prior
images. The 6.50 GB reduction comes from a smaller Aider language environment
and, for Hermes, removing Chromium and its browser helper plus fetching only
the pinned Git commit instead of full repository history. The Aider audit found
no one-off language that could be dropped without removing multiple contested
questions. Virtual sizes include shared layers, and actual incremental disk
cost depends on images already present. Cold build or pull transfer bytes were
not measured.
The default command does not build or pull those images.
