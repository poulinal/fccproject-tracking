
# Higgs → WW

The FCC-ee will operate at center-of-mass energies of 240 and 365 GeV, with integrated luminosities of 10.8 ab⁻¹ and 3 ab⁻¹, respectively. These conditions provide an unprecedented opportunity to study the Higgs boson with high precision. Higgs bosons are produced predominantly via the $e^+e^- \to ZH$ channel, where the Higgs is produced in association with a Z boson. One of the key decay channels to investigate is $H \to WW^*$, which can be analyzed at both center-of-mass energies.

## Challenges

This measurement faces several challenges:
1. **Background Rejection**: Significant backgrounds arise from diboson production ($WW$ and $ZZ$) and single boson processes ($Z+\gamma$).
2. **Irreducible Background**: The $H \to ZZ^*$ decay closely mimics the signal and must be treated either as an irreducible background or simultaneously optimized and fitted.

## Project Objective

The primary goal is to measure the $H \to WW^*$ decay in various sensitive final states and combine the results into a single, precise cross-section measurement. All the different measurements are to be combined into a single uncertainty on the $H \to WW^*$ cross-section.

---

## Preliminary Questions

1. **ZH Production Cross-Section**:  
   The total $ZH$ production cross-section is approximately 200 pb at 240 GeV and 120 pb at 365 GeV. Using the integrated luminosities of 10.8 ab⁻¹ and 3 ab⁻¹, calculate the total number of $ZH$ bosons produced.

2. **Background Events**:  
   Using cross-sections for $WW$ and $ZZ$ processes from [this resource](https://submit.mit.edu/~jaeyserm/fcc/samples/ee_FastSim_winter2023_IDEA.html) (look for `p8_ee_WW` and `p8_ee_ZZ`), calculate the number of expected background events. Compare these to the number of signal events.

3. **Feynman Diagrams**:  
   Draw the Feynman diagrams for the signal ($H \to WW^*$) and key backgrounds ($WW$, $ZZ$, and $Z/\gamma$). Identify key differences to guide event selection and background discrimination.

---

## Decay Channels

Both W and Z bosons are unstable and decay rapidly. Their branching fractions can be found [here](https://en.wikipedia.org/wiki/W_and_Z_bosons). This project will focus on the following final states:

1. $Z(\nu\nu)WW(\ell\nu\ell\nu)$
2. $Z(\nu\nu)WW(qqqq)$
3. $Z(e^+e^-/\mu^+\mu^-)WW(\ell\nu\ell\nu)$
4. $Z(e^+e^-/\mu^+\mu^-)WW(\ell\nu qq)$

These channels will be analyzed for both 240 GeV and 365 GeV. For now, focus on 240 GeV; the analysis for 365 GeV can be extended later.

---

## Analysis Workflow

Choose your preferred channel and proceed as follows:

1. **Signal Event Estimation**:  
   Compute the expected number of signal events, factoring in the branching ratios for the W and Z bosons.

2. **Process Identification**:  
   Identify all signal and background processes relevant to your analysis. A comprehensive list can be found [here](https://submit.mit.edu/~jaeyserm/fcc/samples/ee_FastSim_winter2023_IDEA.html).

3. **Event Validation**:  
   Implement the analysis in RDataFrame. Check whether the number of simulated events matches your computed expectations before applying any selection cuts.

4. **Event Selection Optimization**:  
   Optimize the event selection using basic kinematic variables to improve signal-to-background separation.

5. **Likelihood Fit**:  
   Identify one or two kinematic variables suitable for performing a likelihood fit. Use these to extract the cross-section and quantify its uncertainty.

