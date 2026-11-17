# Notice and attribution

This repository extends **neural-ddft-pcm** by Karnik Ram et al.
(<https://github.com/karnikram/neural-ddft-pcm>), the code accompanying

> K. Ram, J. Dijkman, R. van Roij, J.-W. van de Meent, B. Ensing, M. Welling, D. Cremers,
> *Learned Free-Energy Functionals from Pair-Correlation Matching for Dynamical Density Functional Theory*,
> Phys. Rev. E (2025), doi:10.1103/22fd-ykkb, arXiv:2505.09543.

The original repository does not specify a license. All rights to the original code
remain with its authors. The extensions in this repository are by Jean-Pasqual
Sindermann (2025–2026), developed as an interdisciplinary project at the Technical
University of Munich:
- the underdamped DDFT solver in PhiFlow;
- the hybrid PCM + learned superadiabatic force model and its end-to-end training;
- the evaluation and diagnostic tools.

No license is granted for this repository until it has been agreed with the original
authors. Please contact them before reusing or redistributing the code.

`bd/` contains a copy of [mdext](https://github.com/shankar1729/mdext) (BSD 3-Clause,
see `bd/LICENSE`).

Parts of the code, comments and documentation were written with the help of AI
assistants (Anthropic Claude, Google Gemini).
