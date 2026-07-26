# Wall materials — secondary electron emission (SEE)

SEE-yield coefficients for Hall thruster channel wall materials.
Maxwellian-averaged yield `gamma(Te) = gamma_2plusb * a * Te^b` (Te in eV),
from Goebel & Katz, *Fundamentals of Electric Propulsion* (Wiley/JPL, 2008),
Ch. 7, Table 7-1 and Eq. 7.3-30.

| material         | a     | b     | Γ(2+b) | mono-energetic data source        |
|------------------|-------|-------|--------|-----------------------------------|
| BN               | 0.150 | 0.549 | 1.38   | Bugeat & Koppel, IEPC-1995-035    |
| BNSiO2           | 0.123 | 0.528 | 1.36   | Gascon et al., Phys. Plasmas 2003 |
| Al2O3            | 0.145 | 0.650 | 1.49   | Gascon et al., Phys. Plasmas 2003 |
| stainless_steel  | 0.040 | 0.610 | 1.44   | Goebel & Katz, Table 7-1          |

Each `*.txt` carries the full citation in its header. One file per material,
`key value` lines, `#` comments.
